from typing import Protocol

import torch
from torch import Tensor, nn

from lab.components.attention.utils import create_causal_mask
from lab.components.norm.rms_norm import RMSNorm
from lab.components.positional_embeddings.rope import precompute_freqs_cis

from .config import ModelConfig
from .utils import (
    TransformerBlock,
    TransformerForwardData,
    TransformerMask,
    gate_act_fn_clamp,
)


class TransformerProtocol(Protocol):
    tok_embeddings: nn.Embedding
    layers: nn.ModuleList
    masks: nn.ModuleList
    input_norm: RMSNorm
    output_norm: RMSNorm
    output: nn.Linear

    def _init_weights(self, module: nn.Module) -> None: ...


class Transformer(nn.Module, TransformerProtocol):
    def __init__(
        self, config: ModelConfig, device: torch.device | str = "cuda"
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim).bfloat16()

        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.masks = nn.ModuleList()
        for _ in range(config.n_layers // 2):
            self.masks.append(TransformerMask(config.dim))

        self.input_norm = RMSNorm(config.dim, config.norm_eps)
        self.output_norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

        self.mask = create_causal_mask(config.max_seq_len, device=device)

    def _init_weights(self, module: nn.Module) -> None:
        if self.config.initializer_range is not None:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()  # ty: ignore[call-non-callable]
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        if self.config.zero_init_masks:
            for mask in self.masks:
                mask.linear.weight.data.fill_(0.0)  # type: ignore
                mask.linear.bias.data.fill_(0.0)  # type: ignore

    def forward(
        self, inputs: Tensor, start_pos: int = 0, inference: bool = False
    ) -> tuple[Tensor, TransformerForwardData]:
        _bsz, seqlen = inputs.shape
        h = self.input_norm(self.tok_embeddings(inputs))
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = self.mask
        if inference and len(inputs) != self.config.max_seq_len:
            mask = create_causal_mask(len(inputs), device=h.device, start_pos=start_pos)
        if start_pos != 0:
            mask = create_causal_mask(seqlen, device=h.device, start_pos=start_pos)
        mask = mask.type_as(h)

        masks, gates, hidden_states = [], [], []

        layer: TransformerBlock
        for i, layer in enumerate(self.layers):  # type: ignore
            if i < self.n_layers // 2:
                masks.append(self.masks[i](h) + (masks[-1] if i > 0 else 0))
                gates.append(gate_act_fn_clamp(masks[-1]))
            else:
                gates.append(gate_act_fn_clamp(masks[self.n_layers - (i + 1)]))
            h = layer(h, gates[-1], freqs_cis, mask, start_pos)
            hidden_states.append(h)

        masks = torch.stack(masks, 0).squeeze()
        gates = torch.stack(gates, 0).squeeze()
        hidden_states = torch.stack(hidden_states, 0)

        output = self.output(self.output_norm(h))
        return output, TransformerForwardData(masks, gates, hidden_states)

    @staticmethod
    def from_pretrained(
        logdir: str, step: int, device: torch.device, key: str | None = "model"
    ) -> "Transformer":
        import os

        path = os.path.join(logdir, f"state_step{step}.pt")
        model = Transformer(ModelConfig.from_pretrained(logdir, key), device=device)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=device)["model"]
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"state_step{step}.pt not found at {path}")
        return model.to(device)
