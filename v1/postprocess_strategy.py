import json, re

FEEDBACK_FILE="output/real_feedback.jsonl"
OUT_FILE="output/real_feedback_postprocessed.jsonl"

def postprocess(text: str) -> str:
    # remove few-shot continual
    text = text.split("##")[0]
    text = text.strip()
    texts = text.split("\n")
    texts = [t for t in texts if len(t.split(" ")) > 5]
    text = "\n".join(texts)
    return text

if __name__ == "__main__":
    data = []
    with open(FEEDBACK_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            _data['strategy'] = postprocess(_data['strategy'])
            data.append(_data)

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            fout.write(json.dumps(_data) + "\n")