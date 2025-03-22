# Verl-Tool
An unified and easy-to-extend tool-agent training framework based on verl.

## Installation
```bash
uv sync
uv pip install -e verl
uv pip install vllm==0.7.3
uv pip install flash-attn --no-build-isolation
```

## Training
```bash
bash examples/train/train_gsm8k.sh
```

## Test Tool Servers

```bash
python -m verl_tool.servers.serve # Start the server
python -m verl_tool.servers.tests.test_base # Run the tests
```

## ToDos
- [ ] Add python servers and example training scripts
- [ ] Add browser servers and example training scripts
- [ ] Add VLM servers and example training scripts
- [ ] Custom truncation logic for large observations
- [ ] Add wandb logging statistics
- [ ] Add saving logic for every step's sampling results and observations for inspection

## Contribution to tool libraries
Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g. `base`, `python_code`), the name of tool is exactly the name of the python file that you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

