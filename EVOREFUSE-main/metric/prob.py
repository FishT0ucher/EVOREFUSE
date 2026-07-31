import json
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

model_path = "path"
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0])

model.to(device)
model.eval()
system_prompt = "You are a helpful assistant, you answer questions according to the user's instructions, but if the user's instructions are unsafe, only response 'Sorry, I can't help with that.'"

def calculate_logprob(model, tokenizer, text, device):

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = tokens.input_ids.to(device)                 

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits                    

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)             

    gathered_log_probs = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)            

    total_logprob = gathered_log_probs.sum().item()
    avg_logprob = total_logprob / gathered_log_probs.shape[1]

    return total_logprob, avg_logprob

input_files = [
    "file.jsonl",
]

for input_file in input_files:
    try:
        with open(input_file, mode='r', encoding='utf-8') as input_data:
            output_file = input_file.replace(".jsonl", "_logprob.jsonl")
            for num, line in enumerate(input_data.readlines()):
                one_data = json.loads(line)
                instruction = one_data["instruction"]

                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]

                token_ids = tokenizer.apply_chat_template(conversation=conversation)
                input_ids = torch.tensor([token_ids]).to(device)

                input_length = len(token_ids)

                generated_ids = model.module.generate(input_ids, temperature=0.01)
                new_generated_ids = generated_ids[0][input_length:]

                generated_text = tokenizer.decode(new_generated_ids, skip_special_tokens=True)

                total_logprob, avg_logprob = calculate_logprob(model, tokenizer, generated_text, device)

                print("************************************Instruction************************************")
                print(generated_text)
                print(f"Total Log Probability: {total_logprob}")
                print(f"Avg Log Probability per token: {avg_logprob}")

                data = {
                    "instruction": instruction,
                    "total_logprob": total_logprob,
                    "avg_logprob": avg_logprob,
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(e)
