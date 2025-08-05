import json
from collections import defaultdict
from typing import Dict

import torch
import numpy as np
from scipy.stats import skew, kurtosis


def summarize_tensor(tensor: np.ndarray) -> Dict[str, float]:
    assert isinstance(tensor, np.ndarray), "Input must be a numpy array"

    stats = {
        "mean": float(np.mean(tensor)),
        "std": float(np.std(tensor)),
        "min": float(np.min(tensor)),
        "max": float(np.max(tensor)),
        "sparsity": float(np.mean(tensor == 0)),
        "skewness": float(skew(tensor)),
        "kurtosis": float(kurtosis(tensor)),
    }

    return stats


def summarize_model(model: torch.nn.Module, output_path: str) -> None:
    type_counter = defaultdict(int)
    result = []

    for _, module in model.named_modules():
        if len(list(module.children())) == 0:
            module_type = str(module).split("(")[0].strip()
            idx = type_counter[module_type]
            module_name = f"{module_type}:{idx}"
            type_counter[module_type] += 1
            entry = {
                "name": module_name,
                "type": module.__class__.__name__,
                "has_weight": False,
            }

            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.detach().cpu().numpy()
                entry["has_weight"] = True
                entry["summary"] = summarize_tensor(w)
            else:
                entry["summary"] = {}

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def print_summary(name: str, tensor: np.ndarray):
    stats = summarize_tensor(tensor)
    print(f"\n Stats for {name}:")

    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
