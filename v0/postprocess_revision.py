import json, re

REVISION_FILE="output/real_revision.jsonl"
OUT_FILE="output/real_revision_postprocessed.jsonl"

def postprocess(text: str) -> str:
    # remove few-shot continual
    text = text.split("##")[0]
    text = text.strip()
    text = re.sub("\n", "", text)

    sentences = text.split(".")
    if len(sentences) == 1:
        return ""

    # too long sequence for few-shot
    if text[0] == '<' or text[0] == '-':
        return ""
    
    return text

if __name__ == "__main__":
    data = []
    with open(REVISION_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            _data['revision'] = postprocess(_data['revision'])
            data.append(_data)

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            fout.write(json.dumps(_data) + "\n")