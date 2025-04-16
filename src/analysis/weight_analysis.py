from typing import Dict, List, Optional

import numpy as np
from torch import nn


class WeightAnalyer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def get_weight_stats(
        self, target_layers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        stats = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if target_layers is None or name in target_layers:
                    data = param.detach().cpu().numpy()
                    zero_count = np.sum(np.abs(data) < 1e-6)
                    total_count = data.size
                    sparsity = zero_count / total_count
                    stats[name] = {
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "sparsity": float(sparsity),
                    }

        return stats
