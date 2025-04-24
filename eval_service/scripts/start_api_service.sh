#!/bin/bash
# 1. begin ray server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.ray_serve --host $host --port $port --tool_type "python_code" 2>&1 > /dev/null &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. start api service
model_path="GAIR/ToRL-1.5B"
max_turns=1
api_host="0.0.0.0"
api_port=$(shuf -i 30000-31000 -n 1)

CUDA_VISIBLE_DEVICES=0,1  python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool-server-url $tool_server_url \
    --model-path $model_path \
    --max-turns $max_turns \

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. kill all server
pkill -P -9 $server_pid
kill -9 $kill $server_pid
pkill -P -9 $api_server_pid
kill -9 $kill $api_server_pid
