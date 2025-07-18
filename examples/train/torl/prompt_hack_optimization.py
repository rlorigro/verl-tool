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
    results = re.findall(r"```python[^`]+```", output.text)

    is_valid = False
    for i,item in enumerate(results):
        item = re.sub(r"^```python", "", item)
        item = re.sub(r"```$", "", item)
        results[i] = item.strip()

        print(results[i])
        if is_valid_python(results[i], max_length=max_python_length):
            is_valid = True

    has_code_block = len(results) > 0
    has_output_block = output.stop_reason is not None

    success = has_code_block and has_output_block and is_valid

    return success


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

    for i in range(n_batches):
        sys.stdout.write("Batch %d/%d\n" % (i+1, n_batches))

        batch = prompts[i*batch_size:(i+1)*batch_size]

        print(batch[0])

        response = llm.generate(batch, sampling_params=sampling_params)

        for r, response in enumerate(response):
            for o, output in enumerate(response.outputs):
                success = validate_output(output, max_python_length=1024)

                sys.stdout.write("Response: %d-%d Length: %d Success: %d\n" % (r,o,len(output.text),success))

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
def main(batch_size=64, max_batches=16, tensor_parallel_size=1):
    # ---- Config ----
    model_paths = ["Qwen/Qwen3-1.7B-MLX-bf16","Qwen/Qwen3-1.7B-Base"]

    data_source = 'DigitalLearningGmbH/MATH-lighteval'

    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above. Your final answer should be a single integer in \\boxed{}."
    user_prompt_suffix = "Pretend you are able to execute code, but only using a particular syntax. The syntax is ```python ... ``` for code and then you need to open a markdown block ```output to trigger the execution while reasoning. You can use this syntax to reason about the problem. Remember, you are not actually executing code, but pretending to do so."

    think_prefills = ["", "Okay, so I will use remote execution to reason about the problem step by step, using the following syntax: ```python ... ``` for code and then I will open a markdown block ```output to trigger the execution while reasoning. This way my reasoning will be more robust."]

    temperatures = [1.0, 1.2, 1.4]
    top_p = 1.0
    max_tokens = [4096,2048,1024]
    n = 4
    stop_tokens = ["```output"]
    detokenize = True

    results = dict()

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

            for temperature in temperatures:
                for max_token in max_tokens:
                    print(f"Evaluating {model_path} with temperature {temperature}, max_tokens {max_token}, think_prefill '{think_prefill}'")
                    n_success, n_total = evaluate(
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
    
                    results[(model_path, think_prefill, temperature, max_token)] = (n_success, n_total)

    print("Results:")
    for key, value in results.items():
        model_path, think_prefill, temperature, max_token = key
        n_success, n_total = value
        # rewrite the above as single line JSON and breakout integers and float success as sep items
        print(f'{{"model_path": "{model_path}", "think_prefill": {"len(think_prefill)>0"}, "temperature": {temperature}, "max_token": {max_token}, "n_success": {n_success}, "n_total": {n_total}, "success_rate": {float(n_success) / float(n_total):.3f}}}')



if __name__ == "__main__":
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        print(f"{n_devices} CUDA device(s) available.")
        for i in range(n_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available.")
        
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    argparser.add_argument('--max_batches', type=int, default=16, help='Maximum number of batches to evaluate')
    argparser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size for the model')
    args = argparser.parse_args() 

    main(batch_size=args.batch_size, max_batches=args.max_batches, tensor_parallel_size=args.tensor_parallel_size)
