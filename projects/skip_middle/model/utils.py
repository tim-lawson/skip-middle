from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import _score_mod_signature

from lab.components.attention.grouped_query_attention import GroupedQueryAttention
from lab.components.feed_forward.swiglu import SwiGLU
from lab.components.norm.rms_norm import RMSNorm

from .config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.attention = GroupedQueryAttention(config.dim, config.n_heads, config.n_kv_heads)  # fmt: skip
        self.feed_forward = SwiGLU(config.dim, 4 * config.dim, config.multiple_of, config.ffn_dim_multiplier)  # fmt: skip
        self.pre_attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.pre_feed_forward_norm = RMSNorm(config.dim, config.norm_eps)
        self.post_feed_forward_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        gate: Tensor,
        freqs_cis: Tensor,
        mask,
        start_pos: int = 0,
    ) -> Tensor:
        x = x + gate * self.post_attention_norm(
            self.attention(
                self.pre_attention_norm(x),
                freqs_cis,
                apply_score_mod(mask, gate),
                start_pos,
            )
        )
        return x + gate * self.post_feed_forward_norm(
            self.feed_forward(self.pre_feed_forward_norm(x))
        )


class TransformerMask(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.linear(x))


@dataclass
class TransformerForwardData:
    masks: Tensor
    gates: Tensor
    hidden_states: Tensor


def gate_act_fn_clamp(x: Tensor) -> Tensor:
    return 1 - torch.clamp(x, 0, 1)


def apply_score_mod(mask: Tensor | None, gate: Tensor, eps: float = 1e-6) -> Tensor:
    score_mod = torch.log(gate.clamp(eps, 1)).squeeze(-1)  # b q_idx
    if mask is None:
        return score_mod[:, :, None].unsqueeze(1)  # b h q_idx kv_idx
    mask = mask[None, :, :] + score_mod[:, :, None]  # b q_idx kv_idx
    mask = mask.unsqueeze(1)  # b h q_idx kv_idx
    return mask


def create_score_mod(gate: Tensor, eps: float = 1e-6) -> _score_mod_signature:
    score_mod = torch.log(gate.clamp(eps, 1)).squeeze(-1)  # b q_idx
    return lambda score, b, h, q_idx, kv_idx: score + score_mod[b, q_idx]


def init_weights(
    module: nn.Module, initializer_range: float | None, zero_init_masks: bool
) -> None:
    if initializer_range is not None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()  # ty: ignore[call-non-callable]
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)
    if isinstance(module, TransformerMask) and zero_init_masks:
        module.linear.weight.data.fill_(0.0)
        module.linear.bias.data.fill_(0.0)  # type: ignore
