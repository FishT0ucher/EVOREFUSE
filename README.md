# EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions

## Requirements
```setup
python 3.8.18
numpy 1.12.5
transformer 4.43.1
```

## Run EVOREFUSE
```
python framework/evorefuse.py
```

## Fine-tuning LLMs
You can download Llama-factory for fine-tuning, run:
```
llamafactory-cli train finetune/sft.yaml
llamafactory-cli train finetune/dpo.yaml
```

## Evaluation
To evaluate on diversity and confidence, run:

```
python metric/lexical.py
python metric/longppl.py
python metric/prob.py
```

To evaluate on refusal rates, run:

```
python metric/prr.py
python metric/crr.py
```

## Analysis
To visualize pseudo-malicious instructions, run:

```
python visual/gradient.py
python visual/information_flow.py
```

