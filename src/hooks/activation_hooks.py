from typing import Callable, List, Optional

import torch
from torch import nn

from ..hook_base import BaseHookManager


class ActivationHookManager(BaseHookManager):
    def __init__(self) -> None:
        super().__init__()

    def _get_hook(self, name: str) -> Callable:
        def hook(output: torch.Tensor) -> None:
            self.data[name] = output.detach().cpu()

        return hook

    def register_hooks(
        self, model: nn.Module, target_layers: Optional[List[str]] = None
    ) -> None:
        """
        If target layer is None, then the register hook will return full model
        """
        for name, module in model.named_modules():
            if target_layers is None or name in target_layers:
                handle = module.register_forward_hook(self._get_hook(name))
                self.handles.append(handle)
