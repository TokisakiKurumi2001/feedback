import json
from typing import Dict

def apply_template(inp: str) -> Dict:
    template = (
        "### Input: \n{input}\n\n### Feedback: \n"
    )
    prompt = template.format(input=inp)
    return {'prompt': prompt}

if __name__ == "__main__":
    text = ""
    with open("fewshot_revision.jsonl") as fin:
        for line in fin:
            data = json.loads(line)
            if 'instruction' in data:
                text += data['instruction']
            else:
                text += data['prompt']
                text += data['output']

    with open('question.txt') as fin:
        data = fin.read()

    text += apply_template(data)['prompt']

    with open("prompt.txt", "w+") as fout:
        fout.write(text)