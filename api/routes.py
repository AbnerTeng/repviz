# repviz/api/routes.py
import os
import json
from typing import List

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()


@router.get("/")
def read_root():
    return "Welcome to the RePViz API! Visit /static/index.html to view the dashboard."


@router.get("/api/models")
def list_models() -> List[str]:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))

    if not os.path.exists(root_dir):
        return []

    models = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    return models


@router.get("/api/model-structure")
def get_model_structure(model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/{model_name}/model_structure.json")


@router.get("/api/activations")
def get_activations(model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/{model_name}/activations.json")


@router.get("/api/activations/plots")
def get_activation_scatter(
    model_name: str,
    layer_y: str = Query(..., description="Layer Y-axis"),
):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    act_path = f"{root_dir}/{model_name}/activations.json"

    if not os.path.exists(act_path):
        return JSONResponse(status_code=404, content={"error": "Activations not found"})
    try:
        with open(act_path, "r") as f:
            activations = json.load(f)

        layer_names = list(activations.keys())

        if layer_y not in layer_names:
            return JSONResponse(
                status_code=400,
                content={"error": f"Layer '{layer_y}' not found in activations"},
            )

        select_idx = layer_names.index(layer_y)
        if select_idx < 1:
            return JSONResponse(
                status_code=400,
                content={"error": "No previous layer to compare with"},
            )

        layer_x = layer_names[select_idx - 1]
        scatter_data = {
            "x": np.array(activations[layer_x][0]).mean(axis=0).tolist(),
            "y": np.array(activations[layer_y][0]).mean(axis=0).tolist(),
            "layer_x": layer_x,
            "layer_y": layer_y,
        }
        return JSONResponse(content=scatter_data)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to decode activations JSON"},
        )


@router.get("/api/weights")
def get_weights(model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/{model_name}/weights.json")


@router.get("/api/weights/plots")
def get_weight_histogram(
    model_name: str,
    layer_y: str = Query(..., description="Layer to plot weights for"),
):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    weight_path = f"{root_dir}/{model_name}/weights.json"

    if not os.path.exists(weight_path):
        return JSONResponse(status_code=404, content={"error": "Weights not found"})

    try:
        with open(weight_path, "r") as f:
            weights = json.load(f)

        if layer_y not in weights:
            return JSONResponse(
                status_code=400,
                content={"error": f"Layer '{layer_y}' not found in weights"},
            )
        hist_data = {
            "weights": np.array(weights[layer_y]).flatten().tolist(),
            "layer_y": layer_y,
        }
        return JSONResponse(content=hist_data)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to decode weights JSON"},
        )


@router.get("/api/gradients")
def get_gradients():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/gradients.json")


@router.get("/api/predictions")
def get_predictions():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/predictions.json")


@router.get("/api/cka_similarity")
def get_cka_similarity():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/cka_similarity.json")
