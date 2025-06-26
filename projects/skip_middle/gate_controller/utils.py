from dataclasses import dataclass
from typing import Protocol

import torch
from simple_parsing import Serializable

from lab.utils import layers_to_dict


class GateControllerProtocol(Protocol):
    def __init__(
        self, n_layers: int, device: torch.device | str, **kwargs: dict
    ) -> None: ...

    def loss(self, gates: torch.Tensor) -> torch.Tensor: ...

    def update(self, gates_mean: torch.Tensor, gates_var: torch.Tensor) -> None: ...

    def to_dict(self, prefix: str = "") -> dict[str, float]: ...


def flipcat(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 1, "Expected 1D tensor"
    return torch.cat([x, x.flip(0)])


def make_mean_targets(
    n_layers: int,
    mean_targets: list[float] | str,
    mean_target_start: float | None,
    mean_target_end: float | None,
    device: torch.device | str,
) -> torch.Tensor:
    n_targets = n_layers // 2
    if mean_targets == "auto":
        assert mean_target_start is not None, "Expected target_start"
        assert mean_target_end is not None, "Expected target_end"
        x = torch.linspace(mean_target_start, mean_target_end, n_targets, device=device)
    else:
        if isinstance(mean_targets, str):
            mean_targets = [float(x) for x in mean_targets.split(" ")]
        assert len(mean_targets) == n_targets, f"Expected {n_targets} targets"
        x = torch.tensor(mean_targets, device=device)
    return flipcat(x)


def make_var_targets(
    mean_targets: torch.Tensor, var_target_delta: float
) -> torch.Tensor:
    return (mean_targets * (1 - mean_targets) + var_target_delta).clamp(min=0)


class ExponentialMovingAverage:
    def __init__(self, steps: float) -> None:
        self.beta = 1 - (1 / steps)
        self.initialized = False

    def update(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialized = True
            self.ema = x.clone()
        else:
            self.ema.mul_(self.beta).add_(x * (1 - self.beta))
        return self.ema


DEFAULT_VAR_TARGET_DELTA = 0
DEFAULT_MEAN_COEF_INIT = 0
DEFAULT_VAR_COEF_INIT = 0
DEFAULT_COEF_MAX = +torch.inf
DEFAULT_COEF_MIN = -torch.inf
DEFAULT_EMA_STEPS_SHORT = 1
DEFAULT_EMA_STEPS_LONG = 10
DEFAULT_DELTA_MIN = 0.01


@dataclass
class TargetGateControllerConfig(Serializable):
    mean_targets: list[float] | str = "auto"
    """Target values for the gate means, or "auto" for linearly spaced values."""

    mean_target_start: float | None = None
    """If mean_targets is "auto", the start of the target values for the gate means."""

    mean_target_end: float | None = None
    """If mean_targets is "auto", the end of the target values for the gate means."""

    var_target_delta: float = DEFAULT_VAR_TARGET_DELTA
    """Value added to the target values for the gate variances."""

    mean_coef_init: float = DEFAULT_MEAN_COEF_INIT
    """Initial value of the adaptive coefficients for the gate mean loss."""

    var_coef_init: float = DEFAULT_VAR_COEF_INIT
    """Initial value of the adaptive coefficients for the gate variance loss."""

    coef_max: float = DEFAULT_COEF_MAX
    """Maximum value of the adaptive coefficients."""

    coef_min: float = DEFAULT_COEF_MIN
    """Minimum value of the adaptive coefficients."""

    ema_steps_short: int = DEFAULT_EMA_STEPS_SHORT
    """Timescale in steps for short-term exponential moving averages (EMAs)."""

    ema_steps_long: float = DEFAULT_EMA_STEPS_LONG
    """Timescale in steps for long-term exponential moving averages (EMAs)."""

    delta_min: float = DEFAULT_DELTA_MIN
    """Minimum absolute value above which deltas are significant."""


class TargetGateController(GateControllerProtocol):
    def __init__(
        self,
        n_layers: int,
        config: TargetGateControllerConfig,
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
        loss = torch.zeros(1, device=gates.device)
        loss += (self.mean_coefs * gates.mean((1, 2))).mean()
        loss += (self.var_coefs * gates.var((1, 2))).mean()
        return loss.mean()

    def update(self, gates_mean: torch.Tensor, gates_var: torch.Tensor) -> None:
        mean_deltas = gates_mean - self.mean_targets
        var_deltas = gates_var - self.var_targets

        self.mean_deltas_short.update(mean_deltas)
        self.mean_deltas_long.update(mean_deltas)
        self.var_deltas_short.update(var_deltas)
        self.var_deltas_long.update(var_deltas)

        self.update_mean_coefs(mean_deltas)
        self.mean_coefs.clamp_(self.config.coef_min, self.config.coef_max)

        self.update_var_coefs(var_deltas)
        self.var_coefs.clamp_(self.config.coef_min, self.config.coef_max)

    def update_mean_coefs(self, mean_deltas: torch.Tensor) -> None:
        raise NotImplementedError

    def update_var_coefs(self, var_deltas: torch.Tensor) -> None:
        raise NotImplementedError

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            **layers_to_dict(self.mean_targets, f"{prefix}mean/targets"),
            **layers_to_dict(self.mean_coefs, f"{prefix}mean/coefs"),
            **layers_to_dict(self.mean_deltas_short.ema, f"{prefix}mean/deltas/short"),
            **layers_to_dict(self.mean_deltas_long.ema, f"{prefix}mean/deltas/long"),
            **layers_to_dict(self.var_targets, f"{prefix}var/targets"),
            **layers_to_dict(self.var_coefs, f"{prefix}var/coefs"),
            **layers_to_dict(self.var_deltas_short.ema, f"{prefix}var/deltas/short"),
            **layers_to_dict(self.var_deltas_long.ema, f"{prefix}var/deltas/long"),
        }
