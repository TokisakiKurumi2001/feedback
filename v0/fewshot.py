import json
from typing import Dict, Tuple

def define_task() -> str:
    return (
        "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nProvide feedback for the following homework. The feedback should be in bullet point, each point should be in 5-10 words.\n\n"
    )

def apply_template(inp: str, output: str) -> Dict:
    template = (
        "### Input: \n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
    )
    prompt = template.format(input=inp)
    return {'prompt': prompt, 'output': output + "\n"}

def split_inp_out(text: str) -> Tuple[str, str]:
    inp, out, _ = text.split('\n----\n')
    return inp, out

if __name__ == "__main__":

    with open('sample1.txt') as fin:
        data1 = fin.read()

    with open('sample2.txt') as fin:
        data2 = fin.read()

    inps = []
    outs = []
    i, o = split_inp_out(data1)
    inps.append(i)
    outs.append(o)
    i, o = split_inp_out(data2)
    inps.append(i)
    outs.append(o)

    with open('fewshot.jsonl', 'w+') as fout:
        fout.write(json.dumps({'instruction': define_task()}) + "\n")
        for i, o in zip(inps, outs):
            write_json = apply_template(i, o)
            fout.write(json.dumps(write_json) + "\n")