import os
import re
import sys
import ast
import gc

from vllm import LLM, SamplingParams
import datasets
from transformers import AutoTokenizer
import torch
import argparse
from omegaconf import OmegaConf
import json


"""
Check if the given code is valid and non-trivial Python code.
"""
def is_valid_python(code: str, max_length=1024) -> bool:
    if len(code) > max_length:
        return False
    try:
        tree = ast.parse(code)
        for node in tree.body:
            # Ignore trivial things like empty Expr with Ellipsis or strings
            if isinstance(node, ast.Expr):
                if isinstance(node.value, (ast.Constant)):
                    continue  # trivial expression like "..." or a docstring
            return True  # At least one meaningful node
        return False  # No meaningful statements
    except Exception as e:
        return False 


"""
Check three conditions:
1. There is at least one code block in the output.
2. The code block is valid Python code.
3. There is an output block (stop reason is not None).
If all conditions are met, return True; otherwise, return False.
"""
def validate_output(output: str, max_python_length: int) -> bool:
    results = re.findall(r"```python(.*?)```", output.text, re.DOTALL)

    is_valid = False
    for i,item in enumerate(results):
        # Strip the code block markers so that we can validate the Python code
        item = re.sub(r"^```python", "", item)
        item = re.sub(r"```$", "", item)
        results[i] = item.strip()

        if is_valid_python(results[i], max_length=max_python_length):
            is_valid = True

    # Check if there is at least one code block based on regex results
    has_code_block = len(results) > 0

    # Check if there is an output block, only valid if ```output is the only stop token provided to vLLM`
    has_output_block = output.stop_reason is not None

    success = has_code_block and has_output_block and is_valid

    return success, is_valid, has_code_block, has_output_block


"""
Function to generate a function that processes each example in the dataset.
"""
def make_map_fn(split, data_source, system_prompt=None, user_prompt_suffix=""):
    def process_fn(example, idx):
        question = example.pop('problem') + "\n" + user_prompt_suffix

        data = {
            "data_source": data_source,
            "ability": "math",
            "extra_info": {
                'split': split,
                'index': idx
            }
        }

        if system_prompt is not None:
            data["prompt"] = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        else:
            data["prompt"] = [
                {
                    "role": "user",
                    "content": question
                }
            ]


        return data

    return process_fn


"""
Evaluate a model and a combination of parameters for its ability to generate the correct syntax for python code tool use
"""
def evaluate(
        output_dir,
        output_name,
        data_source,
        tokenizer,
        llm,
        system_prompt,
        user_prompt_suffix,
        temperature,
        top_p,
        max_tokens,
        n,
        stop_tokens,
        detokenize,
        batch_size,
        max_batches=sys.maxsize
):
    # ---- Initialization ----
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    
    train_dataset = dataset['train']
    train_dataset = train_dataset.map(function=make_map_fn(split='train', data_source=data_source, system_prompt=system_prompt, user_prompt_suffix=user_prompt_suffix), with_indices=True)

    print(len(train_dataset), "train items loaded")

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, stop=stop_tokens, detokenize=detokenize)

    # ---- Execution ----
    prompts = [tokenizer.apply_chat_template(x['prompt'], add_generation_prompt=True, tokenize=False) for x in train_dataset]

    n_success = 0
    n_total = 0
    
    n_batches = int(len(train_dataset) / batch_size)
    n_batches = min(n_batches, max_batches)

    out_path = os.path.join(output_dir, output_name + ".jsonl")
    
    with open(out_path, 'w') as f:
        for i in range(n_batches):
            print("Batch %d/%d\n" % (i+1, n_batches))

            batch = prompts[i*batch_size:(i+1)*batch_size]

            response = llm.generate(batch, sampling_params=sampling_params)

            for r, response in enumerate(response):
                for o, output in enumerate(response.outputs):
                    result = validate_output(output, max_python_length=1024)
                    success, is_valid, has_code_block, has_output_block = result

                    print("Response: %d-%d Length: %d Success: %d\n" % (r,o,len(output.text),success))

                    out_json = {
                        "prompt": response.prompt,
                        "response": output.text,
                        "success": success,
                        "is_valid": is_valid,
                        "has_code_block": has_code_block,
                        "has_output_block": has_output_block
                    }

                    # Append to file
                    f.write(json.dumps(out_json) + "\n")
                    n_success += success
                    n_total += 1

            print("Success rate: %.3f" % (float(n_success) / float(n_total)))

    return n_success, n_total


"""
variables
 - model type
 - temperature
 - think hack
 - dataset (math500)
 - max output tokens
 - dspy?
"""
def main(
        output_dir,
        model_paths,
        data_source,
        system_prompt,
        user_prompt_suffixes,
        think_prefills,
        temperatures,
        max_tokens,
        n,
        batch_size=64, 
        max_batches=16, 
        tensor_parallel_size=1
):

    detokenize = True
    top_p = 1.0
    stop_tokens = ["```output"]

    results = dict()

    if os.path.exists(output_dir):
        raise Exception(f"Output directory {output_dir} already exists. Please choose a different directory.")
    
    os.makedirs(output_dir, exist_ok=True)

    # ---- Execution ----
    for m,model_path in enumerate(model_paths):
        if m > 0:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Loading the model... " + model_path, flush=True)
        llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        
        for think_prefill in think_prefills:
            # only initialize model related things once to save time
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            tokenizer.chat_template = "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n" \
            + "<think>\n" + think_prefill \
            + "' }}{% endif %}"

            for user_prompt_suffix in user_prompt_suffixes:
                for temperature in temperatures:
                    for max_token in max_tokens:
                        output_name = '_'.join([re.sub(r'\W+', '_', model_path), "think"+str(int(len(think_prefill)>0)), "suffix"+str(int(len(user_prompt_suffixes)>0)), "temp"+str(temperature), "maxlen"+str(max_token)])
                        
                        print(f"Evaluating: {output_name}", flush=True)

                        n_success, n_total = evaluate(
                            output_dir=output_dir,
                            output_name=output_name,
                            data_source=data_source,
                            tokenizer=tokenizer,
                            llm=llm,
                            system_prompt=system_prompt,
                            user_prompt_suffix=user_prompt_suffix,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_token,
                            n=n,
                            stop_tokens=stop_tokens,
                            detokenize=detokenize,
                            batch_size=batch_size,
                            max_batches=max_batches
                        )
                        
                        print(f"Success rate: {n_success}/{n_total} = {float(n_success) / float(n_total):.3f}")
        
                        results[(model_path, think_prefill, user_prompt_suffix, temperature, max_token)] = (n_success, n_total)

    out_path = os.path.join(output_dir, 'results.jsonl')
    with open(out_path, 'w') as out_file:
        print("Results saved to", out_path)
        for key, value in results.items():
            model_path, think_prefill, user_prompt_suffix, temperature, max_token = key
            n_success, n_total = value

            out_file.write(f'{{"model_path": "{model_path}", "think_prefill": "{len(think_prefill)>0}", "user_prompt_suffix": "{len(user_prompt_suffix)>0}", "temperature": {temperature}, "max_token": {max_token}, "n_success": {n_success}, "n_total": {n_total}, "success_rate": {float(n_success) / float(n_total):.3f}}}\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    args = argparser.parse_args()
    
    config = OmegaConf.load(args.config)

    # check that think prefill and user prompt suffixes are lists with at most 2 items
    if len(config.think_prefills) != 2 or len(config.user_prompt_suffixes) != 2:
        raise ValueError("Both think_prefills and user_prompt_suffixes must be lists with exactly 2 items, one of which is an empty string.")

    if len(config.think_prefills) == 2:
        # check one is empty
        if not (config.think_prefills[0] == "" or config.think_prefills[1] == ""):
            raise ValueError("One of the think_prefills must be an empty string.")
    
    if len(config.user_prompt_suffixes) == 2:
        if not (config.user_prompt_suffixes[0] == "" or config.user_prompt_suffixes[1] == ""):
            raise ValueError("One of the user_prompt_suffixes must be an empty string.")

    main(
        output_dir=config.output_dir,
        model_paths=config.model_paths,
        data_source=config.data_source,
        system_prompt=config.system_prompt,
        user_prompt_suffixes=config.user_prompt_suffixes,
        think_prefills=config.think_prefills,
        temperatures=config.temperatures,
        max_tokens=config.max_tokens,
        n=config.n,
        batch_size=config.batch_size,
        max_batches=config.max_batches,
        tensor_parallel_size=config.tensor_parallel_size
    )


"""
Example config:

output_dir: "/data/prompt_optimization/run1"
batch_size: 64
max_batches: 16
tensor_parallel_size: 1
temperatures: [1.0, 1.2, 1.4]
top_p: 1.0
max_tokens: [4096, 2048, 1024]
n: 4

model_paths:
  - "Qwen/Qwen3-1.7B"
  - "Qwen/Qwen3-1.7B-Base"

data_source: "DigitalLearningGmbH/MATH-lighteval"
system_prompt: "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above. Your final answer should be a single integer in \\boxed{}."
user_prompt_suffixes:
  - ""
  - "Pretend you are able to execute code, but only using a particular syntax. The syntax is ```python ... ``` for code and then you need to open a markdown block ```output to trigger the execution while reasoning. You can use this syntax to reason about the problem. Remember, you are not actually executing code, but pretending to do so."
think_prefills:
  - ""
  - "Okay, so I will use remote execution to reason about the problem step by step, using the following syntax: ```python ... ``` for code and then I will open a markdown block ```output to trigger the execution while reasoning. This way my reasoning will be more robust."
"""
