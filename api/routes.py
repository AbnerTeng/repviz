# repviz/api/routes.py
import os
import json
from typing import List

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse

from tools import cka, gram_linear

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


@router.get("/api/weights")
def get_weights(model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/{model_name}/weights.json")


@router.get("/api/gradients")
def get_gradients(model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/{model_name}/gradients.json")


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


@router.get("/api/gradients/plots")
def get_gradient_histogram(
    model_name: str,
    layer_y: str = Query(..., description="Layer to plot gradients for"),
):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    grad_path = f"{root_dir}/{model_name}/gradients.json"

    if not os.path.exists(grad_path):
        return JSONResponse(status_code=404, content={"error": "Gradients not found"})

    try:
        with open(grad_path, "r") as f:
            gradients = json.load(f)

        if layer_y not in gradients:
            return JSONResponse(
                status_code=400,
                content={"error": f"Layer '{layer_y}' not found in gradients"},
            )
        gradients = np.clip(np.array(gradients[layer_y]).flatten(), 1e-6, 1e6)
        hist_data = {
            "gradients": gradients.tolist(),
            "layer_y": layer_y,
        }
        return JSONResponse(content=hist_data)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to decode gradients JSON"},
        )


@router.get("/api/predictions")
def get_predictions():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/predictions.json")


@router.get("/api/cka-similarity")
def get_cka_similarity(model1: str, model2: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))

    model1_activations_path = os.path.join(root_dir, f"{model1}/activations.json")
    model2_activations_path = os.path.join(root_dir, f"{model2}/activations.json")

    if not os.path.exists(model1_activations_path) or not os.path.exists(
        model2_activations_path
    ):
        return JSONResponse(
            status_code=404,
            content={"error": "One or both model activations not found"},
        )

    try:
        with open(model1_activations_path, "r") as f:
            model1_activations = json.load(f)

        with open(model2_activations_path, "r") as f:
            model2_activations = json.load(f)

        used_activs1 = {k: v for k, v in model1_activations.items() if v}
        used_activs2 = {k: v for k, v in model2_activations.items() if v}
        cka_matrices = np.zeros((len(used_activs1), len(used_activs2)))

        for i, (_, v1) in enumerate(used_activs1.items()):
            for j, (_, v2) in enumerate(used_activs2.items()):
                cka_value = cka(
                    gram_linear(np.array(v1).mean(axis=0)),
                    gram_linear(np.array(v2).mean(axis=0)),
                )
                cka_matrices[len(used_activs1) - i - 1, j] = cka_value
        print(cka_matrices)
        cka_similarity = {
            "model1": model1,
            "model2": model2,
            "cka_similarity": cka_matrices.tolist(),
        }
        return JSONResponse(content=cka_similarity)

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to decode activations JSON"},
        )
