from typing import Callable, List, Optional

import torch
from torch import nn

from ..hook_base import BaseHookManager


class GradientHookManager(BaseHookManager):
    def __init__(self) -> None:
        super().__init__()

    def _get_hook(self, name: str) -> Callable:
        def hook(grad: torch.Tensor) -> None:
            self.data[name] = grad.detach().cpu()

        return hook

    def register_hooks(
        self, model: nn.Module, target_layers: Optional[List[str]] = None
    ) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if target_layers is None or name in target_layers:
                    handle = param.register_hook(self._get_hook(name))
                    self.handles.append(handle)
