from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from simple_parsing import Serializable

from lab.train.optimizers import AdamWOptimizerConfig

from .model import TransformerProtocol


@dataclass
class OptimizerConfig(Serializable):
    default: AdamWOptimizerConfig
    masks: AdamWOptimizerConfig
    norms: AdamWOptimizerConfig


def get_optimizers(
    model: TransformerProtocol, config: OptimizerConfig
) -> torch.optim.Optimizer:
    param_groups = get_param_groups(model)

    adamw_params = [
        {"params": param_groups["default"], **get_adamw_params(config.default)},
        {"params": param_groups["norms"], **get_adamw_params(config.norms)},
        {"params": param_groups["masks"], **get_adamw_params(config.masks)},
    ]
    adamw_optimizer = torch.optim.AdamW(adamw_params, fused=True)

    return adamw_optimizer


def get_param_groups(model: TransformerProtocol) -> dict[str, list[nn.Parameter]]:
    default = [model.tok_embeddings.weight, model.output.weight]  # never muon
    default += [p for n, p in model.layers.named_parameters() if "norm" not in n]

    norms = [model.input_norm.weight, model.output_norm.weight]
    norms += [p for n, p in model.layers.named_parameters() if "norm" in n]  # 1D only

    masks = list(model.masks.parameters())  # 1D only

    return {"default": default, "norms": norms, "masks": masks}  # type: ignore


def get_adamw_params(config: AdamWOptimizerConfig) -> dict[str, Any]:
    return {
        "lr": config.lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
    }
