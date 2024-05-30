import json
from typing import Dict
from loguru import logger

DATA_FILE="output/real_feedback_postprocessed.jsonl"
OUT_FILE="output/strategy_cnn_v2.jsonl"

def apply_template(article: str, output: str, strategy: str) -> Dict[str, str]:
    template = "### Article: {article}\n\nPlot out keypoints and summarize the passage."
    prompt = template.format(article=article)
    template_output = "{strategy}\n\n### Summary: {output}"
    output = template_output.format(strategy=strategy, output=output)
    return {'prompt': prompt, 'output': output}

def apply_template_normal(article: str, output: str, strategy: str) -> Dict[str, str]:
    template = "### Article: {article}\n\nSummarize the passage in 3 sentences."
    prompt = template.format(article=article)
    return {'prompt': prompt, 'output': output}

if __name__ == "__main__":
    data = []
    with open(DATA_FILE, encoding='utf-8') as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)
    logger.info(f"Accumulate {len(data)} samples.")

    with open(OUT_FILE, "w+", encoding="utf-8") as fout:
        for _data in data:
            json_obj = apply_template(**_data)
            fout.write(json.dumps(json_obj) + "\n")

            json_obj = apply_template_normal(**_data)
            fout.write(json.dumps(json_obj) + "\n")