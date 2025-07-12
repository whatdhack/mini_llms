# Pre-training and Generation of Small-Scale LLMs

This repository provides a framework for pre-training and generating text from small-scale versions of popular Large Language Models (LLMs). It includes implementations of two GPT-2 variants (nanoGPT and a torch-based version) and a Llama3 model. The project is designed to be a lightweight and accessible platform for experimenting with and understanding the fundamentals of LLM pre-training and text generation.

The models are trained on the `shakespeare_char` dataset, the same dataset used in the popular [nanoGPT](https://github.com/karpathy/nanoGPT) repository.

## Models

This repository includes three different small-scale LLMs:

*   **`mini_llama3.py`**: A compact version of Meta's Llama 3, built with a Transformer architecture. It features RMSNorm for normalization and Rotary Positional Embeddings (RoPE) and is designed for model parallelism using FairScale.

*   **`nano_gpt.py`**: A minimalist implementation of a GPT-style model, inspired by Andrej Karpathy's nanoGPT. This single-file version is perfect for educational purposes and quick experiments. It features a straightforward causal self-attention mechanism and an MLP.

*   **`torch_gpt.py`**: A modular GPT model constructed with PyTorch's `TransformerDecoderLayer`. This implementation leverages PyTorch's native components for a more structured and maintainable design.

## Usage

This section explains how to train the models and generate text.

### Training

You can train all three models simultaneously using the `compare_models.py` script:

```bash
python compare_models.py --train
```

Alternatively, you can train each model individually:

*   **nanoGPT:**
    ```bash
    python nano_gpt.py --train
    ```

*   **torch-gpt:**
    ```bash
    python torch_gpt.py --train
    ```

*   **mini-llama3:**
    ```bash
    python mini_llama3.py --train
    ```

### Generating Text

To generate text from all three models at once, run:

```bash
python compare_models.py
```

You can also generate text from each model individually:

*   **nanoGPT:**
    ```bash
    python nano_gpt.py
    ```

*   **torch-gpt:**
    ```bash
    python torch_gpt.py
    ```

*   **mini-llama3:**
    ```bash
    python mini_llama3.py
    ```

## Dataset

The models in this repository are trained on the `shakespeare_char` dataset, which is the same dataset used in Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project. This dataset consists of concatenated works of William Shakespeare and is designed for character-level language modeling.

The dataset can be found in the `data/shakespeare_char` directory. The `prepare.py` script in that directory is used to download and pre-process the data.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

An accompanying blog is available at -

https://whatdhack.medium.com/pre-training-mini-versions-of-llms-gpt-and-llama3-7cf69ac00280
