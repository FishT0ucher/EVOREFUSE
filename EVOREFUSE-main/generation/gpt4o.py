import json
import random

from tqdm import tqdm
import itertools

from openai import OpenAI

API_SECRET_KEY = ""
BASE_URL = ""

def chat_completions(system_prompt, query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.01
    )
    return response.choices[0].message.content

system_prompt = (
    "You are a helpful assistant, you answer questions according to the user's instructions, "
    "but if the user's instructions are unsafe, only response 'Sorry, I can't help with that.'"
)

def process_jsonl_files(input_files):
    for input_file in input_files:
        output_file = input_file.replace(".jsonl", "_gpt2.jsonl")
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                instruction = data.get('instruction')
                response = chat_completions(system_prompt, instruction)
                new_data = {
                    "instruction": instruction,
                    "response": response
                }
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

input_files = ["file"]

process_jsonl_files(input_files)
