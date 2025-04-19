# LLM Tool Calling Service

This repository contains a service that enables LLMs to call tools, specifically focusing on Python code execution capabilities.

## Setup and Installation

### 1. Activate the Python Interpreter Tool Server

Start the Python code execution tool server:

```bash
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)  # Randomly select a port
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"
```

> **Important:** Note the exact port chosen by the tool server as you'll need it later.

### 2. Model Preparation

If you need to convert a Megatron model to Hugging Face format:

See the [VERL documentation](https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model) for detailed instructions.

## Architecture

The main tool-calling logic is implemented in: `model_service.py`, which took `manager.py` as a reference.

## Running the Service

### Start the Complete Service

```bash
# Select GPU to use
export CUDA_VISIBLE_DEVICES=7

# Start the tool server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# Start the API service
chmod +x start_service.sh
./start_api_service.sh /path/to/model 0.0.0.0 8000 $tool_server_url
```

### Configuration

Update the tool server URL in `app.py` to match the actual port used:

```python
# Line 94 in app.py
tool_server_url=os.environ.get("TOOL_SERVER_URL", "http://localhost:30150/get_observation")
```

Replace `30150` with the port number that was chosen randomly for your tool server.

## API Usage

Call the API endpoint with the following example:

```bash
curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "acecoder",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to check if a number is prime."
      }
    ]
  }'
```

The service will return a response that may include executed Python code and its results.

## Misc

How to convert megatron/vllm model ckpt to huggingface ckpt:

~~~bash
python3 /data/yi/verl-tool/verl/scripts/model_merger.py --local_dir /data/yi/verl-tool/checkpoints/acecoder/qwen_qwen2.5-7b-instruct-grpo-n8-b4-t0.9/global_step_560/actor
~~~