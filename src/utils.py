from collections import defaultdict
from typing import Any, Dict, Optional, List

from torch import nn


def list_child_modules(
    model: Any, print_out: bool = True
) -> Optional[Dict[str, nn.Module]]:
    child_mod_names = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            child_mod_names[name] = module

    if print_out:
        for name, module in child_mod_names.items():
            print(name, module)
    else:
        return child_mod_names


def list_child_modules_type(model: Any) -> List[str]:
    types = set()

    for _, module in model.named_modules():
        if len(list(module.children())) == 0:
            types.add(type(module).__name__)

    return sorted(list(types))


def group_layers_by_type(model: nn.Module) -> defaultdict:
    grouped = defaultdict(list)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            grouped[type(module).__name__].append(name)

    return grouped
