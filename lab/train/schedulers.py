from dataclasses import dataclass

import torch
from simple_parsing import Serializable


@dataclass
class LinearCosineSchedulerConfig(Serializable):
    warmup_steps: float = 0.1
    start_factor: float = 0.1


def get_linear_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LinearCosineSchedulerConfig,
    train_steps: int,
    cosine_decay: bool = True,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(config.warmup_steps * train_steps)
    if not cosine_decay:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.start_factor, total_iters=warmup_steps
        )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.start_factor, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
        last_epoch=0,
    )
