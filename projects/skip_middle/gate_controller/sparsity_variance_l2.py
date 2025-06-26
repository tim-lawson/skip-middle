from dataclasses import dataclass

import torch
from simple_parsing import Serializable

from lab.utils import l2_norm, layers_to_dict

from .utils import (
    DEFAULT_EMA_STEPS_LONG,
    DEFAULT_EMA_STEPS_SHORT,
    DEFAULT_VAR_TARGET_DELTA,
    ExponentialMovingAverage,
    GateControllerProtocol,
    make_mean_targets,
    make_var_targets,
)


@dataclass
class SparsityVarianceL2GateControllerConfig(Serializable):
    mean_targets: list[float] | str = "auto"
    """Target values for the gate means, or "auto" for linearly spaced values."""

    mean_target_start: float | None = None
    """If mean_targets is "auto", the start of the target values."""

    mean_target_end: float | None = None
    """If mean_targets is "auto", the end of the target values."""

    var_target_delta: float = DEFAULT_VAR_TARGET_DELTA
    """Value added to the target values for the gate variances."""

    mean_coef_init: float = 0.1
    """Initial value of the adaptive coefficients for the gate mean loss."""

    var_coef_init: float = 0.1
    """Initial value of the adaptive coefficients for the gate variance loss."""

    ema_steps_short: float = DEFAULT_EMA_STEPS_SHORT
    """Timescale in steps for short-term exponential moving averages (EMAs)."""

    ema_steps_long: float = DEFAULT_EMA_STEPS_LONG
    """Timescale in steps for long-term exponential moving averages (EMAs)."""


class SparsityVarianceL2GateController(GateControllerProtocol):
    def __init__(
        self,
        n_layers: int,
        config: SparsityVarianceL2GateControllerConfig,
        device: torch.device | str,
    ) -> None:
        self.config = config

        self.mean_coefs = config.mean_coef_init * torch.ones(n_layers, device=device)
        self.var_coefs = config.var_coef_init * torch.ones(n_layers, device=device)

        self.mean_targets = make_mean_targets(
            n_layers,
            config.mean_targets,
            config.mean_target_start,
            config.mean_target_end,
            device,
        )
        self.var_targets = make_var_targets(self.mean_targets, config.var_target_delta)

        self.mean_deltas_short = ExponentialMovingAverage(config.ema_steps_short)
        self.mean_deltas_long = ExponentialMovingAverage(config.ema_steps_long)
        self.var_deltas_short = ExponentialMovingAverage(config.ema_steps_short)
        self.var_deltas_long = ExponentialMovingAverage(config.ema_steps_long)

    def loss(self, gates: torch.Tensor) -> torch.Tensor:
        l2_norms_mean = l2_norm(gates.mean((1, 2)) - self.mean_targets)
        l2_norms_var = l2_norm(gates.var((1, 2)) - self.var_targets)

        loss = torch.zeros(1, device=gates.device)
        loss += (self.mean_coefs * l2_norms_mean).mean()
        loss += (self.var_coefs * l2_norms_var).mean()
        return loss.mean()

    def update(self, gates_mean: torch.Tensor, gates_var: torch.Tensor) -> None:
        mean_deltas = gates_mean - self.mean_targets
        var_deltas = gates_var - self.var_targets

        self.mean_deltas_short.update(mean_deltas)
        self.mean_deltas_long.update(mean_deltas)
        self.var_deltas_short.update(var_deltas)
        self.var_deltas_long.update(var_deltas)

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            **layers_to_dict(self.mean_targets, f"{prefix}mean/targets"),
            **layers_to_dict(self.mean_deltas_short.ema, f"{prefix}mean/deltas/short"),
            **layers_to_dict(self.mean_deltas_long.ema, f"{prefix}mean/deltas/long"),
            **layers_to_dict(self.var_targets, f"{prefix}var/targets"),
            **layers_to_dict(self.var_deltas_short.ema, f"{prefix}var/deltas/short"),
            **layers_to_dict(self.var_deltas_long.ema, f"{prefix}var/deltas/long"),
        }
