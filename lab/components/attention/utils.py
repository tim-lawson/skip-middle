import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


def create_causal_mask(
    seqlen: int, device: torch.device | str, start_pos: int = 0
) -> torch.Tensor:
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask])
    return mask
