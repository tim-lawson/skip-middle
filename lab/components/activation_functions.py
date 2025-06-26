from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

Act = Literal["relu", "gelu", "silu"]

ACT2FN: dict[Act, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}
