from transformers import AutoTokenizer, AutoModelForCausalLM

import json, re
from loguru import logger
from tqdm import tqdm
from loguru import logger
from typing import Dict, Tuple, Optional

FEEDBACK_ONLY=False
NO_SHOTS=2
TASK="summarization"
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
SHOT_SAMPLE="data/shot{idx}.txt"
FEEDBACK_FILE="output/real_feedback_postprocessed.jsonl"
OUTPUT_FILE="output/real_revision.jsonl"

def define_task(have_revision: bool=False) -> str:
    if not have_revision:
        return (
            "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nProvide feedback for the following homework. The feedback should be in bullet point, each point should be in 5-10 words.\n\n"
        )
    else:
        return (
            "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nRewrite the homework according to the feedbacks. The feedback should be in bullet point, each point should be in 5-10 words.\n\n"
        )

def task_template(task: str, data_dict: Dict) -> str:
    if task == 'summarization':
        return data_dict['text']

def apply_template(inp: str, feedback: str, have_revision: bool=False, revision: Optional[str] = None) -> Dict:
    if not have_revision:
        template = (
            "### Input: \n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(input=inp)
        return {'prompt': prompt, 'output': feedback + "\n"}
    else:
        template = (
            "### Input: \n{input}\n\n### Feedback: \n{feedback}\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(input=inp, feedback=feedback)
        return {'prompt': prompt, 'output': revision + "\n"}

def load_shots(no_shots: int = 2, have_revision: bool = False) -> str:
    def split_inp_out(text: str) -> Tuple[str, str, str]:
        inp, fb, revision = text.split('\n----\n')
        return inp, fb, revision

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
        return_dict = apply_template(inp, fb, have_revision, revision)
        return_str += return_dict['prompt']
        return_str += return_dict['output']
    return return_str

def transform(item: Dict, shots: str, have_revision: bool=False) -> str:
    query = task_template(TASK, item)
    if not have_revision:
        template = (
            "### Input: \n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(input=query)
    else:
        fb = item['feedback']
        template = (
            "### Input: \n{input}\n\n### Feedback: \n{feedback}\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(input=query, feedback=fb)
    return shots + prompt

if __name__ == "__main__":
    # read data
    data = []
    with open(FEEDBACK_FILE) as fin:
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
    instruction = define_task(have_revision=(not FEEDBACK_ONLY))
    shots = load_shots(no_shots=NO_SHOTS, have_revision=(not FEEDBACK_ONLY))
    shots = instruction + shots
    logger.success(f"Successfully load {NO_SHOTS} shots")

    # into the loop of tokenizer, generate, accumulate the result
    revisions = []
    for _data in tqdm(data):
        text = transform(_data, shots, have_revision=(not FEEDBACK_ONLY))
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=100)[0]
        prompt_length = inputs['input_ids'].size()[1]
        out = tokenizer.decode(outputs[prompt_length:], skip_special_tokens=True)
        revisions.append(out)

    with open(OUTPUT_FILE, "w+") as fout:
        for _data, revision in zip(data, revisions):
            _data['revision'] = revision
            fout.write(json.dumps(_data) + "\n")
