#!/bin/bash

# 设置默认值
export MODEL_PATH=${MODEL_PATH:-"/data/yi/verl-tool/checkpoints/acecoder/qwen_qwen2.5-7b-instruct-grpo-n8-b4-t0.9/global_step_560/actor/huggingface"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}
export TOOL_SERVER_URL=${TOOL_SERVER_URL:-"http://localhost:30286/get_observation"}
export MAX_TURNS=${MAX_TURNS:-"5"}
export VALID_ACTIONS=${VALID_ACTIONS:-'["python"]'}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.9"}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}


echo "启动LLM工具API服务"
echo "模型路径: $MODEL_PATH"
echo "工具服务器URL: $TOOL_SERVER_URL"

# 启动服务
python app.py