import torch
import numpy as np
from scipy.stats import skew, kurtosis


def summarize_tensor(tensor: torch.Tensor):
    flat = tensor.detach().cpu().view(-1).numpy()

    stats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "sparsity": float(np.mean(flat == 0)),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat)),
    }

    return stats


def print_summary(name: str, tensor: torch.Tensor):
    stats = summarize_tensor(tensor)
    print(f"\n Stats for {name}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
