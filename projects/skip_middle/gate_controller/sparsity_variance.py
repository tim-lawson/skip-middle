from dataclasses import dataclass

import torch
from simple_parsing import Serializable

from lab.utils import layers_to_dict

from .utils import GateControllerProtocol


@dataclass
class SparsityVarianceGateControllerConfig(Serializable):
    mean_coef_init: float = +0.01
    """Initial value of the coefficients for the gate mean loss."""

    var_coef_init: float = -0.01
    """Initial value of the coefficients for the gate variance loss."""


class SparsityVarianceGateController(GateControllerProtocol):
    def __init__(
        self,
        n_layers: int,
        config: SparsityVarianceGateControllerConfig,
        device: torch.device | str,
    ) -> None:
        self.config = config

        self.mean_coefs = config.mean_coef_init * torch.ones(n_layers, device=device)
        self.var_coefs = config.var_coef_init * torch.ones(n_layers, device=device)

    def loss(self, gates: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=gates.device)
        loss += (self.mean_coefs * gates.mean((1, 2))).mean()
        loss += (self.var_coefs * gates.var((1, 2))).mean()
        return loss.mean()

    def update(self, gates_mean: torch.Tensor, gates_var: torch.Tensor) -> None:
        pass

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            **layers_to_dict(self.mean_coefs, f"{prefix}mean/coefs"),
            **layers_to_dict(self.var_coefs, f"{prefix}var/coefs"),
        }
