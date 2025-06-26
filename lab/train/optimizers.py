from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from simple_parsing import Serializable


@dataclass
class AdamWOptimizerConfig(Serializable):
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2


ParamsT = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)


def get_adamw_optimizer(params: ParamsT, config: AdamWOptimizerConfig):
    return torch.optim.AdamW(
        params,
        config.lr,
        (config.beta1, config.beta2),
        config.eps,
        config.weight_decay,
        fused=True,
    )


@dataclass
class MuonOptimizerConfig(Serializable):
    lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    lookahead: float = 0.95
    ns_steps: int = 5
    weight_decay: float = 0.0
