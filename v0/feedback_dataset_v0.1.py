import json
from typing import Dict
from loguru import logger

DATA_FILE="output/data.jsonl"
OPTION=3
PREDICT_FILE="output/real_feedback_postprocessed.jsonl"
OUT_FILE=f"output/fb_data_v{OPTION}.jsonl"

def apply_template(article: str, predict: str, feedback: str, output: str, option: int, **kwargs) -> Dict[str, str]:
    if option == 1:
        template = "### Summary: {summary}\n\n### Feedback: {feedback}\n\nRewrite the summary according to the feedback."
        prompt = template.format(summary=predict, feedback=feedback)
    elif option == 2:
        template = "###Article: {article}\n\n{cmd}"
        prompt = template.format(article=article, cmd=kwargs['cmd'])
    else:
        template = "###Article: {article}\n\n### Feedback: {feedback}\n\nSummarize the passage in 3 sentences according to the feedback."
        prompt = template.format(article=article, feedback=feedback)
    return {'prompt': prompt, 'output': output}

if __name__ == "__main__":
    preds = []
    with open(PREDICT_FILE, encoding='utf-8') as fin:
        for line in fin:
            _data = json.loads(line)
            p = _data['predict']
            preds.append(p)

    data = []
    with open(DATA_FILE, encoding='utf-8') as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)
    logger.info(f"Accumulate {len(data)} samples.")

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for d, p in zip(data, preds):
            json_obj = apply_template(**_data, predict=p, option=OPTION)
            fout.write(json.dumps(json_obj) + "\n")