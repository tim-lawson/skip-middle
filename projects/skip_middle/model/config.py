from dataclasses import dataclass
from typing import Any

from simple_parsing import Serializable


@dataclass
class ModelConfig(Serializable):
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = 4
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    use_scaled_rope: bool = False
    max_seq_len: int = 1024
    initializer_range: float | None = 0.02
    zero_init_masks: bool = True

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @staticmethod
    def from_pretrained(logdir: str, key: str | None = "model") -> "ModelConfig":
        config = load_config(logdir)
        config = config[key] if key is not None else config
        return get_model_config(config)


def load_config(logdir: str) -> Any:
    import json
    import os

    config_path = os.path.join(logdir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def get_model_config(config: Any) -> ModelConfig:
    return ModelConfig(
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        vocab_size=config["vocab_size"],
        multiple_of=config["multiple_of"],
        ffn_dim_multiplier=config["ffn_dim_multiplier"],
        norm_eps=config["norm_eps"],
        rope_theta=config["rope_theta"],
        use_scaled_rope=config["use_scaled_rope"],
        max_seq_len=config["max_seq_len"],
        zero_init_masks=temp_get_zero_init_masks(config),
    )


# TODO: remove when all experiments updated
def temp_get_zero_init_masks(config: Any) -> bool:
    try:
        return config["zero_init_masks"]
    except KeyError:
        return config["mask_zero_init"]
