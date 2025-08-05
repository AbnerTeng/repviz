# repviz/api/routes.py
import os
from typing import List

from fastapi import APIRouter
from fastapi.responses import FileResponse

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
def get_weights():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    return FileResponse(f"{root_dir}/weights.json")


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
