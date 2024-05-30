import json

PROMPT_FILE="data/sft_pred.jsonl"
OUTPUT_FILE="data/real_processed.jsonl"

def preprocess(str):
    article, cmd = str.split("\n\n")
    template = (
        "Question: {cmd}\n{article}\nStuden's answer: "
    ).format(cmd=cmd, article=article)
    return template

if __name__ == "__main__":
    data = []
    with open(PROMPT_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            prompt = preprocess(_data['prompt'])
            text = prompt + _data['predict']
            _data['text'] = text
            data.append(_data)

    with open(OUTPUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            fout.write(json.dumps(_data) + "\n")
