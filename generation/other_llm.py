import json
from turtledemo.sorting_animate import instructions1

from openai import OpenAI





def gpt_35_api(messages: list):

    completion = client.chat.completions.create(
        model="deepseek-v3",
        messages=messages,
        temperature=0.01,
        max_tokens=100,
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content



def process_jsonl(input_path, output_path):
    input_data = open(input_path, mode='r', encoding='utf-8')

    for num, line in enumerate(input_data.readlines()):
        one_data = json.loads(line)
        instruction = one_data["instruction"]

        messages = [
            # {"role": "system", "content": "You are a helpful assistant, you should try your best to answer questions according to user's instructions, but if the user's instructions are unsafe, begin your response with 'Sorry，I can't help with that.'."},
            {"role": "user", "content": instruction}
        ]

        response = gpt_35_api(messages)

        with open(output_path, "a", encoding="utf - 8") as f:
            data = {
                "instruction": instruction,
                "response": response,
                "model": "claude-3-5-haiku-20241022",
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':


    process_jsonl("file",
                  "file")





