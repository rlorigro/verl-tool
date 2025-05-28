import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--max_func_call", default=4, type=int)
    parser.add_argument("--base_url", default="http://0.0.0.0:5000", type=str)
    parser.add_argument("--num_threads", default=64, type=int)
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)

    # shuffle
    if args.shuffle:
        random.shuffle(examples, seed=datetime.now().timestamp())

    # select start and end
    examples = examples[args.start:len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    # out_file = f'{args.output_dir}/{model_name}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    out_file = f'{args.output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl'
    os.makedirs(f'{args.output_dir}/{data_name}', exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [f for f in os.listdir(f"{args.output_dir}/{data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    # print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    return examples, processed_samples, out_file


def setup(args):
    # load model
    client = openai.Client(base_url=args.base_url, api_key="sk-proj-1234567890") # random api key will be fine
    tokenizer = None

    # infer & eval
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(client, tokenizer, data_name, args))
    
    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append({
        "acc": sum([result["acc"] for result in results]) / len(results),
    })
    
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main(client, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print("full_prompt:", full_prompt)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)


    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call

    # stop words TODO: make it more general
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ['cot']:
        stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
    if args.prompt_type in ['pal', 'tool-integrated', 'tora', 'torl', 'qwen-torl', 'tool_math_qwen']:
        stop_words.extend(["\n\n---", "```output"]) 
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(["Instruction", "Response"])
    elif "qwen" in args.prompt_type:
        stop_words.extend(["assistant", "user", "_end", "_start"])
        if args.prompt_type == "pot-qwen-r1":
            stop_words.extend(["</python>"])
    print("Stop words:", stop_words)
    
    def call_vt_model(prompt):
        try:
            response = client.completions.create(
                prompt=prompt,
                model=args.model_name_or_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens_per_call,
                top_p=args.top_p,
                n=1,
                stop=stop_words,
            )
            return response.choices[0].text
        except Exception as e:
            return None

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        results = list(tqdm(executor.map(call_vt_model, input_prompts), total=len(input_prompts), desc="Calling VT Served Model"))
    print(results.count(None), "failed to get response from model")
    end_prompts = [(i, prompt+result) for i, (prompt, result) in enumerate(zip(input_prompts, results))]

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True)

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    
    result_json['time_use_in_second'] = time_use
    result_json['time_use_in_minite'] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
