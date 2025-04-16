# for model efficiency
from typing import Dict, Optional, Tuple, Union

from torch import nn
from ptflops import get_model_complexity_info


def estimate_tflops(
    model: nn.Module, input_res: Tuple[int], batch_size: int = 1
) -> Optional[Dict[str, Union[int, float]]]:
    macs, params = get_model_complexity_info(
        model=model,
        input_res=input_res,
        as_strings=False,
        print_per_layer_stat=False,
    )
    if isinstance(macs, int) and isinstance(params, int):
        total_flops = 2 * macs
        tflops = (total_flops * batch_size) / 1e12

        return {"macs": macs, "params": params, "tflops": tflops}
    else:
        raise TypeError("macs and parameters are not in type integer")
