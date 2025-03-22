## Installation
```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

```bash
bash examples/train/train_gsm8k.sh
```

```bash
python -m verl_tool.servers.serve
python -m verl_tool.servers.tests.test_base
```


