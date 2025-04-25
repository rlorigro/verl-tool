# Verl-Tool

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
VerlTool: An unified and easy-to-extend tool-agent training framework based on verl.
</h3>

---

<!-- <p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p> -->

## Installation
```bash
pip install uv # if not installed
uv sync
git submodule update --init --recursive
source .venv/bin/activate
uv pip install -e verl[vllm]
uv pip install vllm==0.8.3 # we found some memory leaking bugs for vllm==0.8.2, so choose to use 0.8.3 instead
uv pip install flash-attn --no-build-isolation
```


## Training

First prepare the data for training. You can use the provided script to preprocess the data.
```bash
python verl-tool/examples/data_preprocess/acecoder.py # preprocess the data and save
```

### Single Node Training

```bash
ray start --head --dashboard-host=0.0.0.0 # start ray head node
bash examples/train/acecoder/train.sh # train the model
```

### Multi Node Training
1. Head Node
```bash
ray start --head --dashboard-host=0.0.0.0 # start ray head node
bash examples/train/acecoder/train.sh # train the model
```
2. Worker Node
```bash
ray start --address='head_node_ip:6379' # start ray worker node
# no need to run the training script on worker node
```

## Evaluation
We do all the evaluation by serving the model in an openai compatible api way. See [eval_service/README.md](./eval_service/README.md) for more details.

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

## Contribution
### Contribution to tool libraries
Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g. `base`, `python_code`), the name of tool is exactly the name of the python file that you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

### New reward manager
Go to the [./verl_tool/agent_workers/reward_manager](./verl_tool/agent_workers/reward_manager) directory. Adding your new reward manager. import it in the `__init__.py` file and in `verl_tool/trainer/main_ppo.py` file add your reward manager importing logic.
