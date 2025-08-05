import numpy as np

from .inference import run_inference
from .models import FFN, FFN2
from .registry import Registry

if __name__ == "__main__":
    data = np.load("repviz/notebooks/test.npy")[:100]
    model = FFN(data.shape[1], 7)
    model2 = FFN2(data.shape[1], 7)
    reg = Registry()
    reg.register_model([model, model2])
    run_inference(reg, data, "cpu")
    print("Inference completed and outputs saved.")
