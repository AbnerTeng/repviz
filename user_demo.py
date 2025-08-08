import numpy as np
import torch

from .inference import run_inference
from .models import FFN, FFN2, TinyTabularAttentionModel
from .registry import Registry


if __name__ == "__main__":
    data = np.load("repviz/notebooks/test.npy")[:64]
    label = np.load("repviz/notebooks/test_y.npy")[:64] - 1
    model = FFN(data.shape[1], 7)
    model.load_state_dict(
        torch.load("repviz/model/ffn_covtype_v1.pth", map_location=torch.device("cpu"))
    )
    model2 = FFN2(data.shape[1], 7)
    model2.load_state_dict(
        torch.load("repviz/model/ffn_covtype_v2.pth", map_location=torch.device("cpu"))
    )
    reg = Registry()
    reg.register_model([model, model2])
    run_inference(reg, data, label, "cpu", get_gradients=True)
    print("Inference completed and outputs saved.")
