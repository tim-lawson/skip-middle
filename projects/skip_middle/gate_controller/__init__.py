import torch
from simple_parsing import subgroups

from .adaptive import AdaptiveGateController, AdaptiveGateControllerConfig
from .dummy import DummyGateController, DummyGateControllerConfig
from .proportional import ProportionalGateController, ProportionalGateControllerConfig
from .proportional_derivative import (
    ProportionalDerivativeGateController,
    ProportionalDerivativeGateControllerConfig,
)
from .sparsity import SparsityGateController, SparsityGateControllerConfig
from .sparsity_variance import (
    SparsityVarianceGateController,
    SparsityVarianceGateControllerConfig,
)
from .sparsity_variance_l2 import (
    SparsityVarianceL2GateController,
    SparsityVarianceL2GateControllerConfig,
)
from .utils import GateControllerProtocol

gate_controller_config = subgroups(
    {
        "dummy": DummyGateControllerConfig,
        "adaptive": AdaptiveGateControllerConfig,
        "proportional": ProportionalGateControllerConfig,
        "proportional_derivative": ProportionalDerivativeGateControllerConfig,
        "sparsity": SparsityGateControllerConfig,
        "sparsity_variance": SparsityVarianceGateControllerConfig,
        "sparsity_variance_l2": SparsityVarianceL2GateControllerConfig,
    },  # type: ignore
    default_factory=DummyGateControllerConfig,  # type: ignore
)

GateControllerConfig = (
    DummyGateControllerConfig
    | AdaptiveGateControllerConfig
    | ProportionalGateControllerConfig
    | ProportionalDerivativeGateControllerConfig
    | SparsityGateControllerConfig
    | SparsityVarianceGateControllerConfig
    | SparsityVarianceL2GateControllerConfig
)


def get_gate_controller(
    config: GateControllerConfig, n_layers: int, device: torch.device | str
) -> GateControllerProtocol:
    if isinstance(config, DummyGateControllerConfig):
        return DummyGateController(n_layers, config, device)
    if isinstance(config, AdaptiveGateControllerConfig):
        return AdaptiveGateController(n_layers, config, device)
    if isinstance(config, ProportionalGateControllerConfig):
        return ProportionalGateController(n_layers, config, device)
    if isinstance(config, ProportionalDerivativeGateControllerConfig):
        return ProportionalDerivativeGateController(n_layers, config, device)
    if isinstance(config, SparsityGateControllerConfig):
        return SparsityGateController(n_layers, config, device)
    if isinstance(config, SparsityVarianceGateControllerConfig):
        return SparsityVarianceGateController(n_layers, config, device)
    if isinstance(config, SparsityVarianceL2GateControllerConfig):
        return SparsityVarianceL2GateController(n_layers, config, device)
    raise ValueError(f"Unknown gate controller: {config}")


__all__ = [
    "AdaptiveGateController",
    "AdaptiveGateControllerConfig",
    "DummyGateController",
    "DummyGateControllerConfig",
    "GateControllerConfig",
    "ProportionalDerivativeGateController",
    "ProportionalDerivativeGateControllerConfig",
    "ProportionalGateController",
    "ProportionalGateControllerConfig",
    "SparsityGateController",
    "SparsityGateControllerConfig",
    "SparsityVarianceGateController",
    "SparsityVarianceGateControllerConfig",
    "SparsityVarianceL2GateController",
    "SparsityVarianceL2GateControllerConfig",
    "gate_controller_config",
    "get_gate_controller",
]
