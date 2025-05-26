import io
from typing import Any

import numpy as np
import streamlit as st
import torch


def upload_data() -> Any:
    uploaded_file = st.file_uploader(
        "Choose a file", type=["json", "csv", "txt", "npy"], accept_multiple_files=False
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npy"):
            array = np.load(uploaded_file, allow_pickle=True)
            st.write(array)

            return array

    else:
        st.warning("Please upload a file.")
        return None


def upload_pytorch_model() -> Any:
    uploaded_model = st.file_uploader("Upload PyTorch Model", type=["pt", "pth"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.write(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        st.warning("CUDA is not available. Using CPU.")

    if uploaded_model is not None:
        try:
            buffer = uploaded_model.read()
            state_dict = torch.load(
                io.BytesIO(buffer), map_location=device, weights_only=True
            )
            return state_dict
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
