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
import pickle

from nano_gpt import NanoGPT, GPTConfig, ADAMWConfig, get_batch_real, estimate_loss
from torch_gpt import TorchGPT
from mini_llama3 import Llama
from mini_llama3 import ADAMWConfig as ADAMWConfig_l3
from mini_llama3 import estimate_loss as estimate_loss_l3

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="test modules")
args=parser.parse_args()

def count_nzgrad(optim):
    tcount = 0
    nzcount = 0
    pcount = 0
    totalp  = 0
    for pg in optim.param_groups:
        for p in pg['params']:
            pcount += 1
            totalp += p.numel()
            if p.grad is not None:
                tcount +=  p.grad.numel()
                nzelems = int(p.grad.count_nonzero())
                nzcount += nzelems
            print ( p.shape, nzelems, p.numel(), totalp)

    return pcount, tcount, nzcount


if __name__ == '__main__':
    model_name='compare'
    print(f"Running {model_name}")
    gptconf = GPTConfig(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=False, dataset="shakespeare_char", batch_size=64, device='cuda')
    #gptconf = GPTConfig(block_size=256, vocab_size=65, n_layer=1, n_head=1, n_embd=384, dropout=0.2, bias=False, dataset="shakespeare_char", batch_size=1, device='cuda')
    model_args = dict(n_layer=gptconf.n_layer, n_head=gptconf.n_head, n_embd=gptconf.n_embd, block_size=gptconf.block_size, bias=gptconf.bias, vocab_size=gptconf.vocab_size, dropout=gptconf.dropout) # start with model_args from command line
    gptconf.data_dir = os.path.join('data', gptconf.dataset)

    modelnano = NanoGPT(gptconf)
    modeltorch = TorchGPT(gptconf)
    # mini llama3
    os.environ["WORLD_SIZE"]=str(1)
    os.environ["MASTER_PORT"] =str(8888)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"]=str(0)
    ckpt_dir= "data/llama3"
    tokenizer_path = ckpt_dir+"/tokenizer.model"
    max_seq_len = gptconf.block_size
    max_batch_size = gptconf.batch_size

    generator  = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    modell3 = generator.model
    print (f"modelnano :  {modelnano} ")
    print (f"modeltorch :  {modeltorch} ")
    print (f"modelllama3 :  {modell3} ")
    modelnano.to(gptconf.device)
    modeltorch.to(gptconf.device)
    modell3.to(gptconf.device)
    out_dir = 'out'
    ckpt_name = f'{gptconf.dataset}_{model_name}_ckpt.pt'
    losses_name = f'{gptconf.dataset}.{model_name}.losses.csv'

    if args.test:
        B = Block(gptconf)
        x = torch.ones((1,256,384))
        y = B(x)
        sys.exit(0)

    if args.train:
        print(f"Training  {model_name}")
        modelnano.train()
        modeltorch.train()
        modell3.train()
        adamwconf = ADAMWConfig()
        optimizernano = modelnano.configure_optimizers(adamwconf, gptconf.device_type)
        optimizertorch = modeltorch.configure_optimizers(adamwconf, gptconf.device_type)
        optimizerl3 = modell3.configure_optimizers(ADAMWConfig_l3(), gptconf.device_type)
        get_batch = get_batch_real
        X,Y = get_batch('train', gptconf)
        X1,Y1 = X.detach().clone(), Y.detach().clone()
        X2,Y2 = X.detach().clone(), Y.detach().clone()
        eval_interval = 100
        best_val_loss = 1e9
        schemadf = {'iter':'int', 'model':'str', 'train':'float', 'val':'float'}
        lossdf = pd.DataFrame ( columns = schemadf.keys()).astype(schemadf)
        last_positive = 0
        for bidx in range (2000):
            logits, loss   = modelnano(X, Y)
            logitstorch, losstorch = modeltorch(X1, Y1)
            logitsl3, lossl3 = modell3(X2, 0, Y2)
            last_positive = bidx if (loss -losstorch ) > 0 else last_positive
            print (f"iter {bidx} logits {logits.shape} {logitstorch.shape }  loss {loss:.4f} {losstorch:.4f} {lossl3:.4f} vs nano {(loss-losstorch):0.4f} {last_positive} {(loss-lossl3):0.4f} ")
            X,Y = get_batch('train', gptconf)
            X1,Y1 = X.detach().clone(), Y.detach().clone()
            X2,Y2 = X.detach().clone(), Y.detach().clone()
            loss.backward()
            losstorch.backward()
            lossl3.backward()
            #print (f"   nzgrad nano {count_nzgrad(optimizernano)}, torch {count_nzgrad(optimizertorch)}, l3 {count_nzgrad(optimizerl3)} ")
            #print (f"   l3 {count_nzgrad(optimizerl3)} ")
            optimizernano.step()
            optimizertorch.step()
            optimizerl3.step()
            optimizernano.zero_grad()
            optimizertorch.zero_grad()
            optimizerl3.zero_grad()
            # evaluate the loss on train/val sets and write checkpoints
            if bidx % eval_interval == 0 and bidx > 0:
                lossesnano = estimate_loss(gptconf, modelnano, get_batch)
                print(f"nano  step {bidx}: train loss {lossesnano['train']:.4f}, val loss {lossesnano['val']:.4f}")
                lossdf.loc[len(lossdf)]= (bidx, "nano",  float(lossesnano['train']), float(lossesnano['val']) )
                lossestorch = estimate_loss(gptconf, modeltorch, get_batch)
                print(f"torch step {bidx}: train loss {lossestorch['train']:.4f}, val loss {lossestorch['val']:.4f}")
                lossdf.loc[len(lossdf)]= (bidx, "torch",  float(lossestorch['train']), float(lossestorch['val']) )
                lossesl3 = estimate_loss_l3(gptconf, modell3, get_batch)
                print(f"llama3  step {bidx}: train loss {lossesl3['train']:.4f}, val loss {lossesl3['val']:.4f}")
                lossdf.loc[len(lossdf)]= (bidx, "llama3",  float(lossesl3['train']), float(lossesl3['val']) )
                lossdf.to_csv(os.path.join(out_dir, losses_name), index=False, compression=None)
                for model, losses in [(modelnano, lossesnano), ( modeltorch, lossestorch), (modell3, lossesl3)]:
                    if losses['val'] < best_val_loss: # or always_save_checkpoint:
                        best_val_loss = losses['val']
                        if bidx > 0:
                            checkpoint = {
                                'model': model.state_dict(),
                                'model_name': model.model_name,
                                #'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': bidx,
                                'best_val_loss': best_val_loss,
                                'config': gptconf,
                            }
                            print(f"saving checkpoint to {out_dir}")
                            ckpt_name = f'{gptconf.dataset}_compare_{model.model_name}_ckpt.pt'
                            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
        print ( "training completed ")
    else:
        print("Generating -  ")
        for model in [modelnano, modeltorch, modell3]:
            model_name  = model.model_name
            checkpoint = torch.load(os.path.join(out_dir, f'{gptconf.dataset}_compare_{model_name}_ckpt.pt'))
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
            print ( f"=============== Generating with model {model.model_name} =====================")
            with torch.no_grad():
                with ctx:
                    for k in range(num_samples):
                        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        print(decode(y[0].tolist()))
                        print('---------------')
        print ( "generation completed")
