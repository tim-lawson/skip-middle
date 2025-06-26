from dataclasses import dataclass, field

from simple_parsing import ArgumentGenerationMode, Serializable, parse

from lab.train.data import DataConfig
from lab.train.optimizers import AdamWOptimizerConfig
from lab.train.schedulers import LinearCosineSchedulerConfig
from projects.skip_middle.gate_controller import (
    GateControllerConfig,
    gate_controller_config,
)
from projects.skip_middle.model import ModelConfig, OptimizerConfig
from projects.skip_middle.train import TrainConfig, train

adamw_optimizer_config = AdamWOptimizerConfig(
    lr=1e-3, beta1=0.8, beta2=0.95, eps=1e-10, weight_decay=0
)


@dataclass
class FineWebTrainConfig(TrainConfig, Serializable):
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            train_files="data/fineweb_10B_gpt2/fineweb_train_*.bin",
            train_tokens=None,
            val_files="data/fineweb_10B_gpt2/fineweb_val_*.bin",
            val_tokens=None,
            device_batch_size=32,
            batch_size=512,
        )
    )

    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            default=adamw_optimizer_config,
            masks=adamw_optimizer_config,
            norms=adamw_optimizer_config,
        )
    )

    scheduler: LinearCosineSchedulerConfig = field(
        default_factory=lambda: LinearCosineSchedulerConfig()
    )

    gates: GateControllerConfig = gate_controller_config

    project: str = "skip_middle"
    logdir: str = "logs/skip_middle"


if __name__ == "__main__":
    train(
        parse(
            FineWebTrainConfig, argument_generation_mode=ArgumentGenerationMode.NESTED
        )
    )
