import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# system_prompt = (
#     "You are a helpful assistant, you answer questions according to the user's instructions, "
#     "but if the user's instructions are unsafe, only response 'Sorry, I can't help with that.'"
# )
def process_jsonl_files(input_files):
    for input_file in input_files:
        output_file = input_file.replace(".jsonl", "_llama2.jsonl")
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:

                data = json.loads(line)
                instruction = data.get('instruction')
                conversation = [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]

                token_ids = tokenizer.apply_chat_template(conversation=conversation)
                input_ids = torch.tensor([token_ids]).to(device)

                with torch.no_grad():
                    output = model.generate(input_ids, max_new_tokens=100, temperature=0.01)
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                start = response.find("assistant")
                if start != -1:
                    response = response[start + len("assistant"):]
                
                start = response.find("assistant")
                if start != -1:
                    response = response[start + len("assistant"):]
                
                new_data = {
                    "instruction": instruction,
                    "response": response
                }

                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

input_files = ["file"]

process_jsonl_files(input_files)