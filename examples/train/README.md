# Model Training Scripts

This folder contain scripts for calling the `verl-tool` pipeline for training models with tool-calling capability. The training configuration can be found in [ppo_trainer.yaml](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/trainer/config/ppo_trainer.yaml) and [config.py](https://github.com/TIGER-AI-Lab/verl-tool/blob/dev/train/verl_tool/llm_agent/config.py)

Specifically, `acecoder` is used for training tool-calling coding models. `torl` is used to train tool-calling mathematical models. Other folders are currently under development.

|Model  Name   |Tool            |Task Type|Link|
|--------------|----------------|---------|----|
|Verl-Tool-Math|Code Interpreter|Math     | [`torl`](./torl)   |
|Verl-Tool-Code|Code Interpreter|Code     |  [`acecoder`](./acecoder)  |


