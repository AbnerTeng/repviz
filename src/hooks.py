from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


class HookManager:
    def __init__(self, track_all: bool = False) -> None:
        self.track_all = track_all
        self.hooks = []
        self.activations = defaultdict(list) if track_all else {}

    def _hook_fn(self, name: str) -> Callable:
        def hook(module: nn.Module, input, output):
            out = output.detach().cpu()

            if self.track_all:
                self.activations[name].append(out)
            else:
                self.activations[name] = out

        return hook

    def register_hooks(
        self,
        model: nn.Module,
        partial_matches: Optional[List[str]] = None,
    ) -> None:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                if partial_matches is not None:
                    for partial_match in partial_matches:
                        if partial_match and partial_match in str(module):
                            hook = module.register_forward_hook(self._hook_fn(name))
                            self.hooks.append(hook)
                else:
                    hook = module.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)

    def register_hooks_by_type(
        self,
        model: nn.Module,
        module_types: List[Any],
    ) -> None:
        for name, module in model.named_modules():
            for module_type in module_types:
                if isinstance(module, module_type):
                    hook = module.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)

    def clear_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

        self.hooks = []

    def get_activations(self) -> Dict:
        if self.track_all:
            return {k: torch.stack(v) for k, v in self.activations.items()}

        return self.activations
