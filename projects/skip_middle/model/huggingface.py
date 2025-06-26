from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.hub import PushToHubMixin

from lab.train.utils import get_device

from .config import ModelConfig, get_model_config
from .model import Transformer


class SkipMiddleConfig(PretrainedConfig):
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = 4
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = True
    max_seq_len: int = 1024
    initializer_range: float = 0.02
    zero_init_masks: bool = True

    def __getitem__(self, x):
        return getattr(self, x)


class SkipMiddleModel(PreTrainedModel, GenerationMixin, PushToHubMixin):
    config_class = SkipMiddleConfig

    def __init__(self, config: SkipMiddleConfig, device: torch.device | str) -> None:
        super().__init__(config)
        self.config = config
        self.model = Transformer(get_model_config(config), device)

    def _init_weights(self, module: nn.Module) -> None:
        self.model._init_weights(module)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> CausalLMOutput:
        logits, _ = self.model(inputs=input_ids, inference=True)
        return CausalLMOutput(logits=logits)


if __name__ == "__main__":
    from simple_parsing import Serializable, parse

    @dataclass
    class Args(Serializable):
        logdir: str
        step: int
        repo_id: str

    args = parse(Args)
    device = get_device()

    try:
        config = ModelConfig.from_pretrained(args.logdir, key=None)
    except KeyError:
        config = ModelConfig.from_pretrained(args.logdir, key="model")

    model = SkipMiddleModel(SkipMiddleConfig(**config.to_dict()), device)
    try:
        model.model = Transformer.from_pretrained(
            args.logdir, args.step, device, key=None
        )
    except KeyError:
        model.model = Transformer.from_pretrained(
            args.logdir, args.step, device, key="model"
        )
    model.push_to_hub(repo_id=args.repo_id)  # type: ignore
