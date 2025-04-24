# LLM Tool Calling Service

## Overview
This package provides a service that enables LLMs to call tools, temporarily focusing on Python code execution capabilities.

The service can be accessed via OpenAI's `client.chat.completions.create` API.
Server is managed by `app.py`, while the main tool-calling logic is implemented in `model_service.py`.

## Setup and Installation

### 1. Activate Service

Start the Python code execution tool server and API Service:

```bash
bash eval_service/scripts/start_api_service.sh
```

> You can set your own params in `start_api_service.sh`

### 2. Test the API Service

**Please Replace with your local server address**

```bash
python eval_service/test/test_api.py
```

## Misc

How to convert megatron/vllm model ckpt to huggingface ckpt:

~~~bash
python3 /data/yi/verl-tool/verl/scripts/model_merger.py --local_dir /data/yi/verl-tool/checkpoints/acecoder/qwen_qwen2.5-7b-instruct-grpo-n8-b4-t0.9/global_step_560/actor
~~~