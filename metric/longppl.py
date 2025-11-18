import json
import torch
from transformers import LlamaForCausalLM, AutoTokenizer


def calculate_token_probabilities(model, tokenizer, instruction, threshold):

    inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, :-1, :]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    input_ids = inputs.input_ids[0][1:]
    token_log_probs = []
    valid_log_probs = []
    for i in range(len(input_ids)):
        token_id = input_ids[i].item()
        log_prob = log_probs[i, token_id].item()
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        token_log_probs.append((token, log_prob))
        if log_prob > threshold:
            valid_log_probs.append(log_prob)
    if len(valid_log_probs) == 0:
        num = 0
        valid_log_probs.append(num)
    if valid_log_probs:
        average_log_prob = -sum(valid_log_probs) / len(valid_log_probs)
    else:
        average_log_prob = None

    return token_log_probs, average_log_prob


def process_file(model, tokenizer, input_file_path, output_file_path, threshold):
    with open(input_file_path, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            instruction = data.get('instruction')
            if instruction is not None:
                instruction = str(instruction)
                token_log_probs, average_log_prob = calculate_token_probabilities(model, tokenizer, instruction,
                                                                                  threshold)
                result = {
                    "instruction": instruction,
                    "token_log_probs": token_log_probs,
                    "average_log_prob": average_log_prob
                }
                f_out.write(json.dumps(result) + '\n')


def main(model_path, input_file_paths, output_file_paths, threshold):

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        print("CUDA is not available, using CPU.")
        device = "cpu"

    model = LlamaForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    for input_file, output_file in zip(input_file_paths, output_file_paths):
        process_file(model, tokenizer, input_file, output_file, threshold)


if __name__ == "__main__":
    model_path = "path"
    input_file_paths = ["file.jsonl"]
    output_file_paths = ["file.jsonl"]
    threshold = -2.8
    main(model_path, input_file_paths, output_file_paths, threshold)