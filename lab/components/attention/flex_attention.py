import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    _score_mod_signature,
    create_block_mask,
    flex_attention,
)

from lab.components.attention.utils import repeat_kv
from lab.components.positional_embeddings.rope import apply_rotary_emb

causal_mask_mod: _mask_mod_signature = (  # noqa: E731
    lambda b, h, q_idx, kv_idx: q_idx >= kv_idx
)


def create_block_mask_(
    mask_mod: _mask_mod_signature, seqlen: int, device: str
) -> BlockMask:
    return create_block_mask(mask_mod, None, None, seqlen, seqlen, device)


def create_causal_block_mask(seqlen: int, device: torch.device | str) -> BlockMask:
    return create_block_mask_(causal_mask_mod, seqlen, str(device))


class FlexAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        model_parallel_size: int = 1,
    ) -> None:
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        block_mask: BlockMask | None,
        start_pos: int = 0,
        score_mod: _score_mod_signature | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = flex_attention(xq, xk, xv, score_mod, block_mask, enable_gqa=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # type: ignore

        return self.wo(output)
