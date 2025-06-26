from dataclasses import dataclass

import torch

from .utils import TargetGateController, TargetGateControllerConfig


@dataclass
class ProportionalDerivativeGateControllerConfig(TargetGateControllerConfig):
    coef_update_multiplier: float = 1e-2
    """Multiplier for proportional updates to the adaptive coefficients."""

    deriv_min: float = 0
    """Minimum absolute value above which changes in deltas are significant."""


class ProportionalDerivativeGateController(TargetGateController):
    def __init__(
        self,
        n_layers: int,
        config: ProportionalDerivativeGateControllerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__(n_layers, config, device)
        self.config = config

    def update_mean_coefs(self, mean_deltas: torch.Tensor) -> None:
        delta_mask = mean_deltas.abs() > self.config.delta_min
        derivs = self.mean_deltas_short.ema - self.mean_deltas_long.ema
        deriv_mask = (mean_deltas * derivs) > self.config.deriv_min
        mask = (delta_mask & deriv_mask).float()
        self.mean_coefs.add_(self.config.coef_update_multiplier * mask * mean_deltas)

    def update_var_coefs(self, var_deltas: torch.Tensor) -> None:
        delta_mask = var_deltas.abs() > self.config.delta_min
        derivs = self.var_deltas_short.ema - self.var_deltas_long.ema
        deriv_mask = (var_deltas * derivs) > self.config.deriv_min
        mask = (delta_mask & deriv_mask).float()
        self.var_coefs.add_(self.config.coef_update_multiplier * mask * var_deltas)
