import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "path"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

                                                                                                                                                                                                     

def process_jsonl_files(input_files):
    for input_file in input_files:

        output_file = input_file.replace(".jsonl", "_mistral2.jsonl")
        print(f"Processing file {input_file} and writing to {output_file}")
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, start=1):
                try:
                    data = json.loads(line)
                    instruction = data.get('instruction')

                    full_prompt = instruction

                    input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

                    with torch.no_grad():
                        output = model.generate(input_ids, temperature=0.01)

                    input_length = input_ids.shape[1]
                    new_output_ids = output[0][input_length:]

                    response = tokenizer.decode(new_output_ids, skip_special_tokens=True).strip()
                    new_data = {
                        "instruction": instruction,
                        "response": response
                    }

                    outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"Error processing line {line_num} in file {input_file}: {e}")


input_files = ["file"]


if __name__ == "__main__":
    process_jsonl_files(input_files)