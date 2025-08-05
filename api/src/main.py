from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torchvista import trace_model
from model.ffn import FFN

app = FastAPI()


class ModelInfo(BaseModel):
    features: list


@app.post("/model_detail/")
async def get_model_detail(model_info: ModelInfo):
    # This is a placeholder for model loading.
    # In a real application, you would load your model based on the request.
    model = FFN(len(model_info.features), 2)

    # Generate the model graph
    graph = trace_model(model, torch.tensor(model_info.features))

    # Extract layer statistics
    layer_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            stats = {}
            if module.weight is not None:
                stats["weight"] = {
                    "mean": module.weight.data.mean().item(),
                    "std": module.weight.data.std().item(),
                    "max": module.weight.data.max().item(),
                    "min": module.weight.data.min().item(),
                }
            if module.bias is not None:
                stats["bias"] = {
                    "mean": module.bias.data.mean().item(),
                    "std": module.bias.data.std().item(),
                    "max": module.bias.data.max().item(),
                    "min": module.bias.data.min().item(),
                }
            layer_stats[name] = stats

    return {"graph": graph.to_html(), "layer_stats": layer_stats}
