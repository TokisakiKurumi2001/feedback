import json, re

FEEDBACK_FILE="output/real_feedback.jsonl"
OUT_FILE="output/real_feedback_postprocessed.jsonl"

def have_less_than_n_character(char: str, n: int, text: str) -> bool:
    cnt = 0
    for c in text:
        if c == char:
            cnt += 1
    return cnt < n

def postprocess(text: str) -> str:
    if text[0] != '-':
        text = '- ' + text
    # remove few-shot continual
    text = text.split("##")[0]
    text = text.strip()
    fb_items = text.split("- ")
    # sometimes the feedback is very long, it is actually the error in generation
    fb_items = [""] + [item if item[-1] == "\n" else item + "\n" for item in fb_items if len(item) > 2 and len(item.split(" ")) < 50 and have_less_than_n_character('\n', 2, item)]
    text = "- ".join(fb_items)

    # remove some non-ascii character
    text = re.sub("\u00a0", " ", text)
    text = re.sub('\u2019', "'", text)
    text = re.sub("\u2013", "-", text)

    return text

if __name__ == "__main__":
    data = []
    with open(FEEDBACK_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            _data['feedback'] = postprocess(_data['feedback'])
            data.append(_data)

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            fout.write(json.dumps(_data) + "\n")