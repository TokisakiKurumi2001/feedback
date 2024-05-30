import json
from typing import Dict
from loguru import logger

DATA_FILE="output/real_revision_postprocessed.jsonl"
OUT_FILE="output/fb_data.jsonl"

def apply_template(prompt: str, predict: str, feedback: str, revision: str, use_feedback: bool = True, **kwargs) -> Dict[str, str]:
    if not use_feedback:
        template = (
            "###\nArticle: {passage}\n\nSummarize the above article in 3 sentences."
        )
        prompt = template.format(passage=prompt)
        response = revision
        return {'prompt': prompt, 'output': response}
    else:
        template = (
            "Revise the following summary according to the feedback.\n\n### Summary: {summary}\n\n### Feedback: {feedback}\n\n### Revision: "
        )
        prompt = template.format(summary=predict, feedback=feedback)
        response = revision
        return {'prompt': prompt, 'output': response}

if __name__ == "__main__":
    data = []
    with open(DATA_FILE, encoding='utf-8') as fin:
        for line in fin:
            _data = json.loads(line)
            if _data['revision'] == "":
                continue
            data.append(_data)
    logger.info(f"Accumulate {len(data)} samples.")

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            json_obj = apply_template(**_data)
            fout.write(json.dumps(json_obj) + "\n")
            json_obj = apply_template(**_data, use_feedback=False)
            fout.write(json.dumps(json_obj) + "\n")