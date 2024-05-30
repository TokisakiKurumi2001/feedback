from transformers import AutoTokenizer, AutoModelForCausalLM

import json, re
from loguru import logger
from tqdm import tqdm
from loguru import logger
from typing import Dict, Tuple, Optional

NO_SHOTS=2
TASK="summarization"
PROMPT_FILE="data/processed.jsonl"
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
SHOT_SAMPLE="data/shot{idx}.txt"
OUTPUT_FILE="output/real_feedback.jsonl"

def define_task() -> str:
    return (
        "<|im_start|>You are an English teacher. You will help your student plot out strategy to identify key points and summarize the passage.\n\n"
    )

def apply_template(inp: str, feedback: str, revision: Optional[str] = None) -> Dict:
    template = (
        "### Article: \n{input}\n\n### Strategy: \n<|im_end|><|im_start|>Assistant:"
    )
    prompt = template.format(input=inp)
    return {'prompt': prompt, 'output': feedback + "\n"}

def split_inp_out(text: str) -> Tuple[str, str, str]:
    inp, fb, revision = text.split('\n----\n')
    return inp, fb, revision

def load_shots(no_shots: int = 2) -> str:
    data = []
    for i in range(no_shots):
        idx = i + 1
        filename = SHOT_SAMPLE.format(idx=idx)
        with open(filename) as fin:
            text = fin.read()
            data.append(text)

    return_str = ""
    for text in data:
        inp, fb, revision = split_inp_out(text)
        return_dict = apply_template(inp, fb, revision)
        return_str += return_dict['prompt']
        return_str += return_dict['output']
    return return_str

def transform(item: Dict, shots: str) -> str:
    query = item['article']
    template = (
        "### Article: \n{input}\n\n### Strategy: \n<|im_end|><|im_start|>Assistant:"
    )
    prompt = template.format(input=query)
    return shots + prompt

if __name__ == "__main__":
    # read data
    data = []
    with open(PROMPT_FILE, encoding="utf-8") as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)

    logger.success(f"Read {len(data)} samples")

    # loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
    device = model.device
    logger.success(f"Successfully load {MODEL_PATH} tokenizer and model.")

    # load shots
    instruction = define_task()
    shots = load_shots(no_shots=NO_SHOTS)
    shots = instruction + shots
    logger.success(f"Successfully load {NO_SHOTS} shots")

    # into the loop of tokenizer, generate, accumulate the result
    feedbacks = []
    for _data in tqdm(data):
        text = transform(_data, shots)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=100)[0]
        prompt_length = inputs['input_ids'].size()[1]
        out = tokenizer.decode(outputs[prompt_length:], skip_special_tokens=True)
        feedbacks.append(out)

    with open(OUTPUT_FILE, "w+", encoding="utf-8") as fout:
        for _data, fb in zip(data, feedbacks):
            _data['strategy'] = fb
            fout.write(json.dumps(_data) + "\n")
