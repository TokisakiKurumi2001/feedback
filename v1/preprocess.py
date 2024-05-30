import json

PROMPT_FILE="data/keep.jsonl"
OUTPUT_FILE="data/processed.jsonl"

def preprocess(str):
    article, cmd = str.split("\n\n")
    return article, cmd

if __name__ == "__main__":
    data = []
    with open(PROMPT_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            article, _ = preprocess(_data.pop('prompt'))
            _data['article'] = article
            data.append(_data)

    with open(OUTPUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            fout.write(json.dumps(_data) + "\n")
