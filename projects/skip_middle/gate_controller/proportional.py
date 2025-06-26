from dataclasses import dataclass

import torch

from .utils import TargetGateController, TargetGateControllerConfig


@dataclass
class ProportionalGateControllerConfig(TargetGateControllerConfig):
    coef_update_multiplier: float = 1e-3
    """Multiplier for proportional updates to the adaptive coefficients."""


class ProportionalGateController(TargetGateController):
    def __init__(
        self,
        n_layers: int,
        config: ProportionalGateControllerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__(n_layers, config, device)
        self.config = config

    def update_mean_coefs(self, mean_deltas: torch.Tensor) -> None:
        delta_mask = (mean_deltas.abs() > self.config.delta_min).float()
        self.mean_coefs.add_(self.config.coef_update_multiplier * delta_mask * mean_deltas)  # fmt: skip

    def update_var_coefs(self, var_deltas: torch.Tensor) -> None:
        delta_mask = (var_deltas.abs() > self.config.delta_min).float()
        self.var_coefs.add_(self.config.coef_update_multiplier * delta_mask * var_deltas)  # fmt: skip
