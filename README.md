# General-Reasoner-Zero


🪡 We introduce a novel framework incorporating generative model-based rewards within GRPO, demonstrating substantial improvements in generalization, robustness, and scalability relative to traditional binary rule-based rewards across diverse domains. 

✅ Model-based rewards outperform pattern-based binary verifications in less-structured domains;<br>
✅ Small 14B models achieve robust cross-domain rewards; It boosts MMLU-Pro performance by 15%.<br>
✅ Our method does not require any additional SFT.


## Installation

```
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -e ./verl
pip install vllm==0.8.3
pip install flashinfer-python
pip install math-verify
```


## Usage

```
# prepare data
python data_preprocess.py --local-dir <data_dir>/general-reasoner-data-preview

# prepare verifier
huggingface download TIGER-Lab/general-reasoner-verifier-preview --local-dir <data_dir>/general-reasoner-verifier-preview

# prepare backbone
huggingface download Qwen/Qwen2.5-7B --local-dir <data_dir>/Qwen2.5-7B
```