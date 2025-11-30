# EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal

**Xiaorui Wu**<sup>1</sup>, **Fei Li**<sup>1</sup>, **Xiaofeng Mao**<sup>2</sup>, **Xin Zhang**<sup>3*</sup>, **Li Zheng**<sup>1</sup>, **Yuxiang Peng**<sup>1</sup>, **Chong Teng**<sup>1</sup>, **Donghong Ji**<sup>1*</sup>, **Zhuang Li**<sup>4†</sup>

<sup>1</sup>Wuhan University, <sup>2</sup>Ant Group, <sup>3</sup>Ant International, <sup>4</sup>RMIT University

Large language models (LLMs) often refuse to answer pseudo-malicious instructions, meaning queries that are semantically harmless but still trigger refusals because safety alignment is overly conservative. EVOREFUSE is a prompt optimization approach that generates diverse pseudo-malicious instructions by maximizing the Evidence Lower Bound (ELBO) of the model’s refusal probability.

## 🛠️ Requirements

Ensure your environment is set up with the following dependencies. We recommend using a virtual environment (Conda/Virtualenv).

```bash
# Python Version
python == 3.8.18

# Core Dependencies
numpy == 1.12.5
transformers == 4.43.1
```

## 🚀 Quick Start

To run the main evolutionary algorithm to generate pseudo-malicious instructions:

```bash
python framework/evorefuse.py
```

This process employs an evolutionary algorithm (Mutation, Recombination, and Fitness Evaluation) to evolve seed instructions into high-confidence refusal triggers.

## 🔧 Fine-tuning LLMs

We utilize `Llama-factory` for alignment tasks. You can fine-tune models using the **EVOREFUSE-ALIGN** dataset to mitigate over-refusal while maintaining safety.

**Supervised Fine-Tuning (SFT):**

```bash
llamafactory-cli train finetune/sft.yaml
```

**Direct Preference Optimization (DPO):**

```bash
llamafactory-cli train finetune/dpo.yaml
```

## 📊 Evaluation

Evaluate the generated instructions on diversity, confidence, and refusal rates.

### Diversity & Confidence Metrics

Measure lexical diversity (MSTTR, HDD, MTLD) and model confidence (Log-Prob, LongPPL).

```bash
python metric/lexical.py
python metric/longppl.py
python metric/prob.py
```

### Refusal Rates

Calculate the Prefix Refusal Rate (PRR) and Classifier Refusal Rate (CRR).

```bash
python metric/prr.py
python metric/crr.py
```

## 🔍 Analysis & Visualization

Analyze the underlying causes of over-refusal using gradient-based attribution and information flow.

```bash
# Visualize gradient-based token attribution
python visual/gradient.py

# Analyze information flow across transformer layers
python visual/information_flow.py
```

## 📝 Citation

If you find this code or dataset useful, please cite our **NeurIPS 2025** paper:

```bibtex
@inproceedings{wu2025evorefuse,
  title     = {EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of {LLM} Over-Refusal to Pseudo-Malicious Instructions},
  author    = {Wu, Xiaorui and Li, Fei and Mao, Xiaofeng and Zhang, Xin and Zheng, Li and Peng, Yuxiang and Teng, Chong and Ji, Donghong and Li, Zhuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025}
}
```

-----

### Acknowledgments

This work was supported by Ant Group and Wuhan University Joint Research Program on Large Language Model Safety Alignment.
