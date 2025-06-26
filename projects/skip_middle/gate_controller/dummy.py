from dataclasses import dataclass

import torch

from .utils import GateControllerProtocol


@dataclass
class DummyGateControllerConfig:
    pass


class DummyGateController(GateControllerProtocol):
    def __init__(
        self,
        n_layers: int,
        config: DummyGateControllerConfig,
        device: torch.device | str,
    ) -> None:
        pass

    def loss(self, gates: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=gates.device)
        return loss.mean()

    def update(self, gates_mean: torch.Tensor, gates_var: torch.Tensor) -> None:
        pass

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {}
