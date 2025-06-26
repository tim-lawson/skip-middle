# skip-middle

> [!NOTE]
> This repository accompanies the preprint Learning to Skip the Middle Layers of Transformers (<https://arxiv.org/abs/2506.21103>).
> For pre-trained models, see [HuggingFace](https://huggingface.co/collections/tim-lawson/learning-to-skip-the-middle-layers-of-transformers-68355e4a9f7a7fa7f01c415f).

We based the underlying Transformer models the reference implementation of Llama 3 (<https://github.com/meta-llama/llama-models/>).
The key difference relative to Llama 3 is that we used the Sandwich-LN scheme (a.k.a. Peri-LN) instead of Pre-LN.
The training codebase is based on the 'nanoGPT speedrun' repository (<https://github.com/KellerJordan/modded-nanogpt>).

## Training

Download the dataset:

```sh
uv run data/download_fineweb_10B_gpt2.py
```

Train a model:

```sh
python -m projects.skip_middle.train_fineweb ...
python -m torch.distributed.run --standalone --nproc_per_node 4 projects/skip_middle/train_fineweb.py ...
```

See [help.txt](help.txt) for command-line arguments or [config.py](projects/skip_middle/train/config.py) for configuration classes.

## Installation

Install uv:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```sh
uv venv
source .venv/bin/activate
```

Install packages:

```sh
uv pip install -e .
```

Install PyTorch:

```sh
UV_TORCH_BACKEND=auto uv pip install torch
```
