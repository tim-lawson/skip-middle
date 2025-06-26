import torch
import torch.nn.functional as F
from torch import Tensor


def l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.vector_norm(x, ord=2, dim=dim)


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
    )


def layers_to_dict(tensor: Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/mean": tensor.mean().item(),
        **{f"{prefix}/layer_{i:02d}": tensor[i].item() for i in range(tensor.size(0))},
    }
