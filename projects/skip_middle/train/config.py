import os
from dataclasses import dataclass

from simple_parsing import Serializable

from lab.train.data import DataConfig, get_steps
from lab.train.schedulers import LinearCosineSchedulerConfig

from ..gate_controller import GateControllerConfig, gate_controller_config
from ..model import ModelConfig, OptimizerConfig


@dataclass
class TrainConfig(Serializable):
    data: DataConfig
    model: ModelConfig

    optimizer: OptimizerConfig
    scheduler: LinearCosineSchedulerConfig

    gates: GateControllerConfig = gate_controller_config
    gates_zero_eps: float = 1e-8

    seed: int = 0

    project: str = "default"
    run_id: str | None = None
    logdir: str = "logs/default"
    log_gradients: bool = False
    log_params: bool = False

    log_every_steps: int = 1
    val_every_steps: int = 100
    save_every_steps: int = -1

    @property
    def batch_tokens(self) -> int:
        return self.data.batch_size * self.model.max_seq_len

    @property
    def minibatch_tokens(self) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return world_size * self.data.device_batch_size * self.model.max_seq_len

    @property
    def train_steps(self) -> int:
        return get_steps(
            self.data.train_files,
            self.data.device_batch_size,
            self.model.max_seq_len,
            self.batch_tokens,
        )

    @property
    def val_steps(self) -> int:
        return get_steps(
            self.data.val_files,
            self.data.device_batch_size,
            self.model.max_seq_len,
            self.minibatch_tokens,
        )

    @property
    def grad_acc_steps(self) -> int:
        return self.batch_tokens // self.minibatch_tokens

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        assert self.train_steps > 0
        assert self.val_steps > 0
        assert self.grad_acc_steps > 0
