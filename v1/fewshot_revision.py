import json
from typing import Dict, Tuple

def define_task() -> str:
    return (
        "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nProvide feedback and rewrite a better revision for the following homework. The feedback should be in bullet point, each point should be in 5-10 words.\n\n"
    )

def apply_template(inp: str, feedback: str, revision: str) -> Dict:
    template = (
        "### Input: \n{input}\n\n### Feedback: \n{feedback}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
    )
    prompt = template.format(input=inp, feedback=feedback)
    return {'prompt': prompt, 'output': revision + "\n"}

def split_inp_out(text: str) -> Tuple[str, str]:
    inp, feedback, revision = text.split('\n----\n')
    return inp, feedback, revision

if __name__ == "__main__":

    with open('sample1.txt') as fin:
        data1 = fin.read()

    with open('sample2.txt') as fin:
        data2 = fin.read()

    inps = []
    fbs = []
    revisions = []
    i, f, r = split_inp_out(data1)
    inps.append(i)
    fbs.append(f)
    revisions.append(r)
    i, f, r = split_inp_out(data2)
    inps.append(i)
    fbs.append(f)
    revisions.append(r)

    with open('fewshot_revision.jsonl', 'w+') as fout:
        fout.write(json.dumps({'instruction': define_task()}) + "\n")
        for i, f, r in zip(inps, fbs, revisions):
            write_json = apply_template(i, f, r)
            fout.write(json.dumps(write_json) + "\n")