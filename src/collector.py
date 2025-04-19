from torch import nn


class Collector:
    def __init__(self) -> None:
        self.weight_snapshots = {}
        self.grad_snapshots = {}

    def capture_weights(self, model: nn.Module, tag: str = "step") -> None:
        self.weight_snapshots[tag] = {
            name: p.detach().cpu() for name, p in model.named_parameters()
        }

    def capture_grads(self, model: nn.Module, tag: str = "step") -> None:
        self.grad_snapshots[tag] = {
            name: p.grad.detach().cpu() if p.grad is not None else None
            for name, p in model.named_parameters()
        }
