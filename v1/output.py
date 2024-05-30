import json
text = ""
with open("fewshot.jsonl") as fin:
    for line in fin:
        data = json.loads(line)
        if 'instruction' in data:
            text += data['instruction']
        else:
            text += data['prompt']
            text += data['output']

with open('fewshot.txt', 'w+') as fout:
    fout.write(text)