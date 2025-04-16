from typing import List, Optional

from torch import nn


class BaseHookManager:
    def __init__(self) -> None:
        self.data = {}
        self.handles = []

    def _get_hook(self, name: str):
        raise NotImplementedError("Subclass must implement _get_hook() method")

    def register_hooks(
        self, model: nn.Module, target_layers: Optional[List[str]] = None
    ) -> None:
        raise NotImplementedError(
            "Subclass must implement register_hooks() method")

    def get_data(self, layer_name: str):
        return self.data.get(layer_name, None)

    def clear(self) -> None:
        self.data.clear()

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
