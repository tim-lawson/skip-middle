import torch
import torch.distributed as dist
from torch import Tensor

from lab.utils import l2_norm


class GateMetrics:
    def __init__(
        self,
        n_layers: int,
        zero_eps: float,
        device: torch.device | str,
        is_distributed: bool = True,
    ) -> None:
        self.n_layers = n_layers
        self.zero_eps = zero_eps

        self.count = 0

        self.masks_mean = torch.zeros(n_layers // 2, device=device)
        self.masks_mean_sq = torch.zeros(n_layers // 2, device=device)

        self.gates_mean = torch.zeros(n_layers, device=device)
        self.gates_mean_sq = torch.zeros(n_layers, device=device)

        self.zeros_mean = torch.zeros(n_layers, device=device)
        self.zeros_mean_sq = torch.zeros(n_layers, device=device)

        self.resid_l2_norm_mean = torch.zeros(n_layers, device=device)
        self.resid_l2_norm_mean_sq = torch.zeros(n_layers, device=device)

        self.is_distributed = is_distributed

    @torch.no_grad()
    def update(self, masks: Tensor, gates: Tensor, hidden_states: Tensor) -> None:
        self.count += 1
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        self.masks_mean += masks.mean((1, 2))
        self.masks_mean_sq += masks.pow(2).mean((1, 2))

        self.gates_mean += gates.mean((1, 2))
        self.gates_mean_sq += gates.pow(2).mean((1, 2))

        zeros = (gates.abs() < self.zero_eps).float()
        self.zeros_mean += zeros.mean((1, 2))
        self.zeros_mean_sq += zeros.pow(2).mean((1, 2))

        resid_l2_norm: Tensor = l2_norm(hidden_states, dim=-1)
        self.resid_l2_norm_mean += resid_l2_norm.mean((1, 2))
        self.resid_l2_norm_mean_sq += resid_l2_norm.pow(2).mean((1, 2))

    # fmt: off
    @torch.no_grad()
    def compute(self) -> None:
        for tensor in [
            self.masks_mean,
            self.masks_mean_sq,
            self.gates_mean,
            self.gates_mean_sq,
            self.zeros_mean,
            self.zeros_mean_sq,
            self.resid_l2_norm_mean,
            self.resid_l2_norm_mean_sq,
        ]:
            if self.is_distributed:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            tensor /= self.count
        self.masks_var = self.masks_mean_sq - self.masks_mean.pow(2)
        self.masks_std = self.masks_var.sqrt()
        self.gates_var = self.gates_mean_sq - self.gates_mean.pow(2)
        self.gates_std = self.gates_var.sqrt()
        self.zeros_var = self.zeros_mean_sq - self.zeros_mean.pow(2)
        self.zeros_std = self.zeros_var.sqrt()
        self.resid_l2_norm_var = self.resid_l2_norm_mean_sq - self.resid_l2_norm_mean.pow(2)
        self.resid_l2_norm_std = self.resid_l2_norm_var.sqrt()

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        metrics = {
            f"{prefix}masks/mean": self.masks_mean.mean().item(),
            f"{prefix}masks/var": self.masks_var.mean().item(),
            f"{prefix}masks/std": self.masks_std.mean().item(),
            f"{prefix}gates/mean": self.gates_mean.mean().item(),
            f"{prefix}gates/var": self.gates_var.mean().item(),
            f"{prefix}gates/std": self.gates_std.mean().item(),
            f"{prefix}zeros/mean": self.zeros_mean.mean().item(),
            f"{prefix}zeros/var": self.zeros_var.mean().item(),
            f"{prefix}zeros/std": self.zeros_std.mean().item(),
            f"{prefix}resid_l2_norm/mean": self.resid_l2_norm_mean.mean().item(),
            f"{prefix}resid_l2_norm/var": self.resid_l2_norm_var.mean().item(),
            f"{prefix}resid_l2_norm/std": self.resid_l2_norm_std.mean().item(),
        }
        for i in range(self.n_layers):
            if i < self.n_layers // 2:
                metrics[f"{prefix}masks/mean/layer_{i:02d}"] = self.masks_mean[i].item()
                metrics[f"{prefix}masks/var/layer_{i:02d}"] = self.masks_var[i].item()
                metrics[f"{prefix}masks/std/layer_{i:02d}"] = self.masks_std[i].item()
            metrics[f"{prefix}gates/mean/layer_{i:02d}"] = self.gates_mean[i].item()
            metrics[f"{prefix}gates/var/layer_{i:02d}"] = self.gates_var[i].item()
            metrics[f"{prefix}gates/std/layer_{i:02d}"] = self.gates_std[i].item()
            metrics[f"{prefix}zeros/mean/layer_{i:02d}"] = self.zeros_mean[i].item()
            metrics[f"{prefix}zeros/var/layer_{i:02d}"] = self.zeros_var[i].item()
            metrics[f"{prefix}zeros/std/layer_{i:02d}"] = self.zeros_std[i].item()
            metrics[f"{prefix}resid_l2_norm/mean/layer_{i:02d}"] = self.resid_l2_norm_mean[i].item()
            metrics[f"{prefix}resid_l2_norm/var/layer_{i:02d}"] = self.resid_l2_norm_var[i].item()
            metrics[f"{prefix}resid_l2_norm/std/layer_{i:02d}"] = self.resid_l2_norm_std[i].item()
        return metrics
    # fmt: on
