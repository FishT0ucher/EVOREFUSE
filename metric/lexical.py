from lexical_diversity import lex_div as ld

import json

def load_instructions_from_jsonl(file_path):
    instructions = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            if 'instruction' in item:

                instructions += item['instruction'] + "\n"
    return instructions



file_name = ["file"]

file_prefix = "path"
for file_path in file_name:
    file_path = file_prefix + file_path
    all_instructions = load_instructions_from_jsonl(file_path)
    # print(all_instructions)
    flt = ld.flemmatize(all_instructions)
    # print(file_path)
    # print(ld.ttr(flt))

    # print(file_path)
    # print(ld.root_ttr(flt))


    print(file_path)
    print(ld.msttr(flt,window_length=800))



