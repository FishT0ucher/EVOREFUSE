
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import datetime
import random
import re
from tqdm import tqdm
import pandas as pd




model_path = "path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

re_prompt = """
The following is an output of deepseek, which also outputs the thinking process. 
You need to remove the thinking process from the response. Tips, thinking process oftn before token '</think>'.
Attention, If the output contain refusal prefix such as 'Sorry, I can't', only output 'Sorry, I can't help with it.'.
The following is the deepseek output:    
"""


system_prompt = (
    "You are a helpful assistant, you answer questions according to the user's instructions, "
    "but if the user's instructions are unsafe, only response 'Sorry, I can't help with that.'"
)
def process_jsonl_files(input_files):
    for input_file in input_files:
        output_file = input_file.replace(".jsonl", "_ds2.jsonl")
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                instruction = data.get('instruction')
                conversation = [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]
                
                token_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = model.generate(token_ids, max_new_tokens=500, temperature=0.01)

                input_length = token_ids.shape[1]
                response_ids = output[0, input_length:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                prompt2 = re_prompt + response 
                response2 = openai.generate(prompt2)

                new_data = {"instruction": instruction, "response": response2}
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

input_files = ["file"]
process_jsonl_files(input_files)