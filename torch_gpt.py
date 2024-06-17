import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn  import functional as F
from torch.nn  import MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
import copy
import os
import numpy as np
import pandas as pd
import argparse
import inspect
import math

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="train the model")
args=parser.parse_args()

# adamw optimizer
@dataclass
class ADAMWConfig:
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla

def get_batch_real(split, gptconf):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(gptconf.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(gptconf.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - gptconf.block_size, (gptconf.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+gptconf.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+gptconf.block_size]).astype(np.int64)) for i in ix])
    if gptconf.device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(gptconf.device, non_blocking=True), y.pin_memory().to(gptconf.device, non_blocking=True)
    else:
        x, y = x.to(gptconf.device), y.to(gptconf.device)
    return x, y

def get_batch_synthetic( gptconf):
    X = torch.randint(65, (gptconf.batch_size, gptconf.block_size), dtype=torch.long).to(gptconf.device)
    Y = torch.randint(65, (gptconf.batch_size, gptconf.block_size), dtype=torch.long).to(gptconf.device)
    return X, Y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    batch_size:int = 64
    data_dir:str='data'
    dataset:str='shakespeare_char'
    device_type:str='cuda'
    device:str='cuda'

class GPTDecoder(nn.Module):
    r"""GPTDecoder is a stack of N decoder layers. Derived for PyTorch's TransformerDecoder.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask,
                         #memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         #memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         #memory_is_causal=memory_is_causal,
                         )

        if self.norm is not None:
            output = self.norm(output)

        return output

class GPTDecoderLayer(nn.Module):
    r"""GPTDecoderLayer is made up of self-attn, and feedforward network. Derived from PyTorch's TransformerDecoderLayer

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True,
                 bias: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=True, **factory_kwargs)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout, inplace=False)
        self.norm_first = norm_first

        # Implementation of Feedforward model
        self.ff_norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, elementwise_affine=True, **factory_kwargs)
        self.ff_linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.ff_dropout2 = Dropout(dropout, inplace=False)
        self.ff_linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.ff_dropout3 = Dropout(dropout, inplace=False)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        #memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        #memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        #memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.ff_norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.ff_norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)


    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        #x = self.ff_linear2(self.ff_dropout2(self.activation(self.ff_linear1(x))))
        x = self.ff_linear2(self.activation(self.ff_linear1(x)))
        return self.ff_dropout3(x)

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class TorchGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = "torch_gpt"
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        decoder_layer = GPTDecoderLayer(d_model=config.n_embd, nhead=config.n_head, dropout=config.dropout, dim_feedforward=4*config.n_embd, activation='gelu', batch_first=True, norm_first=True, bias=False)
        #self.memory = torch.zeros(config.batch_size, config.vocab_size, config.n_embd)

        self.gptnet = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = GPTDecoder(decoder_layer, num_layers=config.n_layer),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        ))
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.gptnet.wte.weight = self.gptnet.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        self.print_params_info()
        print(f"number of parameters: {self.get_num_params():,}")

    #def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    def configure_optimizers(self, optconf,  device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': optconf.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=optconf.learning_rate, betas=(optconf.beta1,optconf.beta2), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.gptnet.wpe.weight.numel()
        return n_params

    def print_params_info(self, ):
        """
        Prints parameter names and sizes  in the model.
        """
        paramsn = [p[0] for p in self.named_parameters()]
        paramss = [p[1].numel() for p in self.named_parameters()]
        paramsh = [p[1].shape for p in self.named_parameters()]
        print ( paramsn )
        print ( paramss )
        print ( paramsh )
        for p in self.named_parameters():
            print ( p[0], p[1].shape, p[1].numel(), )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.gptnet.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.gptnet.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.gptnet.drop(tok_emb + pos_emb)
        tgt_mask   = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        x = self.gptnet.h(x,x, tgt_mask=tgt_mask, tgt_is_causal=True)
        x = self.gptnet.ln_f(x)
        #x = self.gptnet.lm_head(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.gptnet.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.gptnet.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(gptconf, model, get_batch):
    out = {}
    model.eval()
    eval_iters = 200
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, gptconf)
            if True: #with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':

    model_name='torch_gpt'
    print(f"Running {model_name}")

    gptconf = GPTConfig(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=False, dataset="shakespeare_char", batch_size=64, device='cuda')
    model_args = dict(n_layer=gptconf.n_layer, n_head=gptconf.n_head, n_embd=gptconf.n_embd, block_size=gptconf.block_size, bias=gptconf.bias, vocab_size=gptconf.vocab_size, dropout=gptconf.dropout) # start with model_args from command line
    gptconf.data_dir = os.path.join('data', gptconf.dataset)

    ckpt_name = f'{gptconf.dataset}_{model_name}_ckpt.pt'
    losses_name = f'{gptconf.dataset}.{model_name}.losses.csv'

    model = TorchGPT(gptconf).to(gptconf.device)
    print (f"model :  {model}")
    model.to(gptconf.device)

    #X = torch.randint(gptconf.vocab_size, (gptconf.batch_size, gptconf.block_size), dtype=torch.long).to(gptconf.device)
    #logits,loss = model(X)
    #print ( "logits , loss ", logits.shape, loss)


    out_dir = 'out'

    if args.train:
        print(f"Training  {model_name}")
        model.train()
        adamwconf = ADAMWConfig()
        optimizer = model.configure_optimizers(adamwconf, gptconf.device_type)
        get_batch = get_batch_real
        X,Y = get_batch('train', gptconf)
        eval_interval = 100
        best_val_loss = 1e9
        schemadf = {'iter':'int', 'train':'float', 'val':'float'}
        lossdf = pd.DataFrame ( columns = schemadf.keys()).astype(schemadf)
        for bidx in range (1000):
            logits, loss = model(X, Y)
            print (f"iter {bidx} logits {logits.shape}  loss {loss}")
            X,Y = get_batch('train', gptconf)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # evaluate the loss on train/val sets and write checkpoints
            if bidx % eval_interval == 0 and bidx > 0:
                losses = estimate_loss(gptconf, model)
                print(f"step {bidx}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                lossdf.loc[len(lossdf)]= (bidx, float(losses['train']), float(losses['val']) )
                if losses['val'] < best_val_loss: # or always_save_checkpoint:
                    best_val_loss = losses['val']
                    if bidx > 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            #'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': bidx,
                            'best_val_loss': best_val_loss,
                            'config': gptconf,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        #torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
                        lossdf.to_csv(os.path.join(out_dir, losses_name), index=False, compression=None)
        print ( "training completed ")
    else:
        print(f"Generating through {model_name}")
        checkpoint = torch.load(os.path.join(out_dir, ckpt_name))
        model.load_state_dict(checkpoint["model"])
        model.eval()

        init_from = 'resume'
        start='\n'
        num_samples = 10 # number of samples to draw
        max_new_tokens = 500 # number of tokens generated in each sample
        temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
        seed = 1337
        #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        dtype = 'float32'
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in gptconf.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if init_from == 'resume' and 'config' in checkpoint and checkpoint['config'].dataset is not None : # older checkpoints might not have these...
            meta_path = os.path.join('data', checkpoint['config'].dataset, 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)

        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=gptconf.device)[None, ...])
        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    print(decode(y[0].tolist()))
                    print('---------------')
        X,Y = get_batch_synthetic(gptconf)
        Y1 = model(X)
        print ( "generation completed")
