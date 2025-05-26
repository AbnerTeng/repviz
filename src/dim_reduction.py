from typing import Union

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def decomposition(
    mat: Union[torch.Tensor, np.ndarray], n_components: int, method: str = "PCA"
) -> np.ndarray:
    if isinstance(mat, torch.Tensor):
        mat = np.array(mat)

    decomposer = (
        PCA(n_components=n_components)
        if method == "PCA"
        else TSNE(n_components=n_components)
    )
    decomposed_mat = decomposer.fit_transform(mat)

    return decomposed_mat


def plot_umap(activations, layer_name, n_neighbors, n_components):
    umap = UMAP(
        n_neightbors=n_neighbors,
        n_components=n_components,
    ).fit_transform(activations)
