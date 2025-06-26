import os
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DDP:
    rank: int
    world_size: int
    device: torch.device | str
    is_main_process: bool
    is_distributed: bool


def initialize_ddp() -> DDP:
    assert torch.cuda.is_available(), "CUDA is not available"

    if os.environ.get("LOCAL_RANK") is None:
        return DDP(0, 1, "cuda", True, False)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    is_main_process = rank == 0

    return DDP(rank, world_size, device, is_main_process, True)


def initialize_logs(config: dict[str, Any], is_main_process: bool = True):
    assert "logdir" in config, "logdir not in config"
    assert "project" in config, "project not in config"
    assert "run_id" in config, "run_id not in config"

    run = None

    if is_main_process:
        import wandb

        # generate uuid on main process only
        run_id = str(uuid.uuid4()) if config["run_id"] is None else config["run_id"]
        logdir = os.path.join(config["logdir"], run_id)
        os.makedirs(logdir, exist_ok=True)

        logfile = os.path.join(config["logdir"], f"{run_id}.txt")
        with open(logfile, "w", encoding="utf-8") as f:
            f.write(f"torch.version.__version__ {torch.version.__version__}\n")  # type: ignore
            if torch.cuda.is_available():
                f.write(f"torch.version.cuda {torch.version.cuda}\n")  # type: ignore
                f.write(f"{nvidia_smi()}\n")

        run = wandb.init(
            project=config["project"], id=run_id, config=config, save_code=False
        )

    # broadcast logdir (uuid) to all processes
    objs = [logdir] if is_main_process else [None]
    dist.broadcast_object_list(objs, src=0)
    return objs[0] or "", run


def seed_everything(seed: int, cuda: bool = True, cudnn: bool = False) -> None:
    import os
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def nvidia_smi():
    import subprocess

    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
