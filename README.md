Pre-training and generation of small versions of LLMs. 3 models are present - 2 versions of GPT2 -  nanao and torch based, and a Llama3 . The dataset used is the shakespear_char dataset used in NanoGPT ( https://github.com/karpathy/nanoGPT )

To pre-train all 3 model sinultaneously -

python  ./compare_models.py --train

To generate from all 3 models simultaneously

python ./compare_models.py


Each individual model can also be trained and used for generation the same way.

An accompanying blog is available at -

https://whatdhack.medium.com/pre-training-mini-versions-of-llms-gpt-and-llama3-7cf69ac00280
