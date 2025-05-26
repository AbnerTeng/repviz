import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch

from dashboard_utils import upload_data, upload_pytorch_model
from hooks import HookManager
from model.ffn import FFN
from utils import list_child_modules, list_child_modules_type


if __name__ == "__main__":
    if "hook_mgr" not in st.session_state:
        st.session_state.hook_mgr = None
    if "model" not in st.session_state:
        st.session_state.model = None

    st.set_page_config(layout="wide")
    st.title("Repvis: Neural Representation Visualizer")
    uploaded_data = upload_data()
    state_dict = upload_pytorch_model()

    if uploaded_data is not None and state_dict is not None:
        features, label = uploaded_data[:, :-1], uploaded_data[:, -1] - 1
        model = FFN(features.shape[1], len(set(label)))
        model.load_state_dict(state_dict)
        st.session_state.model = model

        with st.expander("Model Structure"):
            st.json(list_child_modules(model, print_out=False))

        hook_layers = st.sidebar.multiselect(
            "Hook Layer Types",
            list_child_modules_type(model),
            default=["LayerNorm"],
        )
        run_hook = st.button("Run Inference & Capture Activations")

        if run_hook:
            hook_mgr = HookManager(track_all=True)
            hook_mgr.register_hooks(model, partial_matches=hook_layers)

            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(features, dtype=torch.float32)[0, :])

            st.session_state.hook_mgr = hook_mgr

    if st.session_state.hook_mgr is not None:
        st.sidebar.title("Layer Inspection")
        selected_layer = st.sidebar.selectbox(
            "Select Layer",
            list(list_child_modules(st.session_state.model, print_out=False).keys()),
        )
        activations = st.session_state.hook_mgr.get_activations()
        data = activations.get(selected_layer, None)

        if data is not None:
            st.subheader("Acitvation Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data.flatten().cpu().numpy(), kde=True, ax=ax)
            st.pyplot(fig)
