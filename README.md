# General-Reasoner: Advancing LLM Reasoning Across All Domains


🪡 We introduce a novel framework incorporating generative model-based rewards within GRPO, demonstrating substantial improvements in generalization, robustness, and scalability relative to traditional binary rule-based rewards across diverse domains. 

✅ Model-based rewards outperform pattern-based binary verifications in less-structured domains;<br>
✅ Small 14B models achieve robust cross-domain rewards; It boosts MMLU-Pro performance by 15%.<br>
✅ Our method does not require any additional SFT.


---

## 🔧 Installation

```bash
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -e ./verl
pip install vllm==0.8.3
pip install flashinfer-python
pip install math-verify
```

---

## 💠 Usage

### 1. Prepare Data
```bash
python data_preprocess.py --local-dir <data_dir>/general-reasoner-data-preview
```

### 2. Download Verifier
```bash
huggingface-cli download TIGER-Lab/general-reasoner-verifier-preview --local-dir <data_dir>/general-reasoner-verifier-preview
```

### 3. Download Backbone Model
```bash
huggingface-cli download Qwen/Qwen2.5-7B --local-dir <data_dir>/Qwen2.5-7B
```

### 4. Configure Training Script
Edit the environment variables in `train_general_reasoner.sh` to fit your system setup.

### 5. Launch Ray Cluster
```bash
ray start --address <MASTER-NODE-IP>:6379
```

### 6. Start Training
```bash
bash train_general_reasoner.sh
```

---

## 🙏 Acknowledgements

This project is built upon the following open-source projects:

- [VERL](https://github.com/volcengine/verl/tree/main/verl)  
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)  
- [simple-evals](https://github.com/openai/simple-evals)

