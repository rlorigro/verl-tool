# LLM Tool Calling Service

## Overview
This package provides a service that enables LLMs to call tools, temporarily focusing on Python code execution capabilities.

The service can be accessed via OpenAI's `client.chat.completions.create` API. When accessing, please ensure the model name corresponds to the one that is being set in the script.

Server is managed by `app.py`, while the main tool-calling logic is implemented in `model_service.py`. `config.py`'s default parameters are overridden in `scripts/start_api_service.sh`.

## Setup and Installation

### 1. Activate Service

Start the Python code execution tool server and API Service:

```bash
bash eval_service/scripts/start_api_service.sh
```

You can set your own params in `start_api_service.sh`. 

Specifically, the parameters are explained as follows:

~~~bash
# set the default host ip for the tool server
host=0.0.0.0

# the tool will randomly pick an available port from 30000 to 31000 when start up
port=$(shuf -i 30000-31000 -n 1)

# set the entry point of the tool server
tool_server_url=http://$host:$port/get_observation

# this is the model path, when calling the tool server please align the model name with this parameter
model_path=Qwen/Qwen2.5-Coder-7B-Instruct

# define the maximum turns for model-tool interaction
max_turns=4

# this is the minimum number of tool-calling activities enforced by the server. When set to a specific number, even if the LLM did not actively asking for tool calling, the tool server will still try to extract Python code from its output.
min_action_num=4

# this is the action token that your LLM shall produce when it is asking for a tool calling round.
action_stop_tokens="<python>"

# Note: num_models * tensor_parallel_size should be equal to the number of GPUs.
# recommend set `num_models` as large as possible to achieve parallel processing.
# tensor_parallel_size: control the tensor sharding across GPUs.
tensor_parallel_size=1
# number of vllm instances.
num_models=8 

# TBD
enable_mtrl=True
~~~

### 2. Test the API Service

**Please Replace with your local server address in the testing `.py` file**

```bash
python eval_service/test/test_api.py
```

## Upload VTModels

```bash
backend=fsdp
checkpoint_path=checkpoints/acecoder/acecoder-fsdp-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-no-tool/global_step_110/actor
hf_upload_path=VerlTool/acecoder-fsdp-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-no-tool-110-step
python3 verl/scripts/model_merger.py --backend $backend --hf_model_path $checkpoint_path/huggingface --hf_upload_path "$hf_upload_path" --local_dir $checkpoint_path --target_dir $checkpoint_path/huggingface

# optional: also upload the step records to the model
step_records_dir=verl_step_records/acecoder-fsdp-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-no-tool
zip -r $step_records_dir/step_records.zip $step_records_dir 
huggingface-cli upload --repo-type model $hf_upload_path $step_records_dir/step_records.zip 

backend=fsdp
checkpoint_path=checkpoints/torl/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6/global_step_330/actor
hf_upload_path=VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6-330-step
python3 verl/scripts/model_merger.py --backend $backend --hf_model_path $checkpoint_path/huggingface --hf_upload_path "$hf_upload_path" --local_dir $checkpoint_path --target_dir $checkpoint_path/huggingface

# optional: also upload the step records to the model
step_records_dir=verl_step_records/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6
zip -r $step_records_dir/step_records.zip $step_records_dir 
huggingface-cli upload --repo-type model $hf_upload_path $step_records_dir/step_records.zip 
```