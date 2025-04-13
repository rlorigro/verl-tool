activate tool (python interpreter) server:

~~~bash
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"
~~~


CONVERT Megatron MODEL to HF:

https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model



Main logic used for detecting tool-call tokens and performing multi-round tool call:
verl-tool/verl_tool/llm_agent/manager.py


RUN: 

export CUDA_VISIBLE_DEVICES=7

host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

chmod +x start_service.sh
./start_api_service.sh /path/to/model 0.0.0.0 8000 $tool_server_url


call API:

curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yi-34b-chat",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to check if a number is prime."
      }
    ]
  }'