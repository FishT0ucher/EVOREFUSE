import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


                   
                                                                                                
                                                                                                 
   


def process_jsonl_files(input_files):
    for input_file in input_files:
        output_file = input_file.replace(".jsonl", "_gemma2.jsonl")

        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line in infile:
                data = json.loads(line.strip())
                instruction = data.get('instruction')

                if not instruction:
                    continue

                full_prompt = instruction

                inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = model.generate(input_ids=inputs['input_ids'], temperature=0.01)

                input_length = inputs.input_ids.shape[1]
                new_output_ids = output_ids[0][input_length:]

                response = tokenizer.decode(new_output_ids, skip_special_tokens=True).strip()

                result = {
                    "instruction": instruction,
                    "response": response
                }

                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

input_files = ["file"]

if __name__ == "__main__":
    process_jsonl_files(input_files)

