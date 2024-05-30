from transformers import AutoTokenizer, AutoModelForCausalLM

import json, re
from loguru import logger
from tqdm import tqdm
from loguru import logger

PROMPT_FILE="prompt.txt"
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"
MODEL_PATH="meta-llama/Llama-2-7b-hf"

if __name__ == "__main__":
    # read data
    with open(PROMPT_FILE) as fin:
        data = fin.read()

    # loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
    device = model.device
    logger.success(f"Successfully load {MODEL_PATH} tokenizer and model.")

    # into the loop of tokenizer, generate, accumulate the result
    text = data
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=100)[0]
    prompt_length = inputs['input_ids'].size()[1]
    out = tokenizer.decode(outputs[prompt_length:], skip_special_tokens=True)

    with open('output.txt', 'w+') as fout:
        fout.write(out)