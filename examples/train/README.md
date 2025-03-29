# AceCoder Training Guide

## Preparation Steps

### 1. Repository Modification
To enable model training with an observation mask, manually modify the `verl` repository located at `verl-tool/verl`:

- Navigate to:
  ```
  verl-tool/examples/train/modified_verl_codes
  ```
- Copy both files and replace their counterparts in:
  ```
  verl-tool/verl/verl/trainer/ppo
  ```

### 2. Dataset Preparation
Download and process the dataset using the following command:
```bash
python examples/data_preprocess/acecoder.py \
  --dataset_path CodeDPO/AceCoderV2-mini-processed \
  --local_dir data/acecoder \
  --add_execution_prompt
```

### 3. Initialize Git Submodules:
run the following commands:
```bash
git submodule init
git submodule update
```

### 3. Logging Configuration
You **MUST** set the Weights & Biases (wandb) key:
```bash
export WANDB_KEY="<your_key>"
```
Alternatively, modify line 65 in `verl-tool/examples/train/train_acecoder.sh`:
- Change:
  ```
  trainer.logger=['console','wandb']
  ```
- To:
  ```
  trainer.logger=['console']
  ```

## Additional Notes

### Data Truncation Adjustment
Due to the following error:
```
NotImplementedError: sequence_length=xxxx is larger than max_length=2048
```

- A data truncation parameter has been added in `train.sh` to handle sequences exceeding the maximum length: `data.truncation='right'`

## Run Experiment
Start training using:
```bash
bash examples/train/train_acecoder.sh
```

