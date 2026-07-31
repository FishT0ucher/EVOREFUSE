import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "path"
input_files = ["file"]

output_suffix = "_qwen2.jsonl"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

                   
                                     
                                           
                                                                                  
   

def build_conversation(instruction):
    return [
                                                       
        {"role": "user", "content": instruction}
    ]

def process_jsonl_files(input_files):
    for input_file in input_files:
        output_file = input_file.replace(".jsonl", output_suffix)

        print(f"Processing {input_file} --> {output_file}")

        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    instruction = data.get('instruction')
                    if not instruction:
                        continue

                    conversation = build_conversation(instruction)

                    input_ids = tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        output_ids = model.generate(input_ids=input_ids, temperature=0.01)

                    new_output_ids = output_ids[0][input_ids.shape[-1]:]

                    response = tokenizer.decode(new_output_ids, skip_special_tokens=True).strip()

                    response = response.split("assistant")[-1].strip()

                    result = {
                        "instruction": instruction,
                        "response": response
                    }

                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

                    print(f"[{line_idx}] Instruction: {instruction}")
                    print(f"Response: {response}\n")

                except Exception as e:
                    print(f"Error processing line {line_idx}: {e}")
                    continue

if __name__ == "__main__":
    process_jsonl_files(input_files)
    print("✅ All files processed.")