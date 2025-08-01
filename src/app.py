import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
from IPython.display import HTML
from torchvista import trace_model

from dashboard_utils import upload_data, upload_pytorch_model
from hooks import HookManager
from model.ffn import FFN
from utils import list_child_modules, list_child_modules_type

sns.set_style("whitegrid")
sns.set_palette("deep")


from monitor import TrainingMonitor

def main():
    if "training_monitor" not in st.session_state:
        st.session_state.training_monitor = None
    if "model" not in st.session_state:
        st.session_state.model = None

    st.set_page_config(layout="wide")
    st.title("Repvis: Neural Representation Visualizer")

    if st.session_state.model is None:
        uploaded_data = upload_data()
        state_dict = upload_pytorch_model()

        if uploaded_data is not None and state_dict is not None:
            features, label = uploaded_data[:, :-1], uploaded_data[:, -1] - 1
            model = FFN(features.shape[1], len(set(label)))
            model.load_state_dict(state_dict)
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.labels = label
    else:
        features = st.session_state.features
        label = st.session_state.labels
        model = st.session_state.model

    if st.session_state.model is not None:
        with st.expander("Model Structure"):
            st.json(list_child_modules(model, print_out=False))

        with st.expander("Model Visualization"):
            graph = trace_model(model, st.session_state.features)
            st.html(graph.to_html(), height=500)

        run_training = st.button("Run Training & Capture Data")

        if run_training:
            training_monitor = TrainingMonitor(model)
            training_monitor.train(features, label)
            st.session_state.training_monitor = training_monitor

    if st.session_state.training_monitor is not None:
        all_data = st.session_state.training_monitor.get_all_data()
        activations = all_data["activations"]
        inputs = all_data["inputs"]
        gradients = all_data["gradients"]
        weights = all_data["weights"]

        st.sidebar.title("Layer Inspection")
        selected_layer = st.sidebar.selectbox(
            "Select Layer",
            list(list_child_modules(st.session_state.model, print_out=False).keys()),
        )
        st.sidebar.title("Analysis")
        analysis_types = st.sidebar.multiselect(
            "Select Analysis",
            ["CKA", "PCA", "UMAP", "LayerNorm"],
        )

        st.header("Layer-wise Analysis")
        
        if "CKA" in analysis_types:
            st.subheader("CKA Similarity Over Time")
            
            cka_over_time = defaultdict(list)
            layer_names = list(activations.keys())

            for i in range(len(activations[selected_layer])):
                for layer_name in layer_names:
                    act1 = activations[selected_layer][i]
                    act2 = activations[layer_name][i]
                    cka_over_time[layer_name].append(cka.linear_cka(act1.view(act1.shape[0], -1), act2.view(act2.shape[0], -1)))
            
            fig, ax = plt.subplots()
            for layer_name, cka_values in cka_over_time.items():
                ax.plot(cka_values, label=layer_name)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("CKA Similarity")
            ax.legend()
            st.pyplot(fig)

        if "LayerNorm" in analysis_types:
            st.subheader("LayerNorm Statistics Over Time")
            stats_over_time = defaultdict(list)
            for i in range(len(activations[selected_layer])):
                stats = summarize_tensor(activations[selected_layer][i])
                for key, value in stats.items():
                    stats_over_time[key].append(value)
            
            fig, ax = plt.subplots(len(stats_over_time), 1, figsize=(10, 20))
            for i, (key, values) in enumerate(stats_over_time.items()):
                ax[i].plot(values)
                ax[i].set_title(key)
            st.pyplot(fig)

        st.header("Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Acitvation Distribution")
            data = activations.get(selected_layer, None)
            if data is not None:
                fig, ax = plt.subplots()
                sns.histplot(data[-1].flatten().cpu().numpy(), kde=True, ax=ax)
                st.pyplot(fig)

        with col2:
            st.subheader("Gradient Distribution")
            grad_data = gradients.get(selected_layer, None)
            if grad_data is not None:
                fig, ax = plt.subplots()
                sns.histplot(grad_data[-1].flatten().cpu().numpy(), kde=True, ax=ax)
                st.pyplot(fig)

        st.header("Weight & Activation Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weight Distribution")
            weight_data = weights.get(selected_layer, None)
            if weight_data is not None:
                fig, ax = plt.subplots()
                sns.histplot(weight_data.flatten().cpu().numpy(), kde=True, ax=ax)
                st.pyplot(fig)

        with col2:
            st.subheader("Activation Effect")
            input_data = inputs.get(selected_layer, None)
            if input_data is not None:
                fig, ax = plt.subplots()
                sns.scatterplot(x=input_data[-1].flatten().cpu().numpy(), y=data[-1].flatten().cpu().numpy(), ax=ax)
                ax.set_xlabel("Input Activations")
                ax.set_ylabel("Output Activations")
                st.pyplot(fig)

        st.header("Dimensionality Reduction")
        col1, col2 = st.columns(2)

        with col1:
            if "PCA" in analysis_types:
                st.subheader("PCA")
                pca_data = dim_reduction.decomposition(activations[selected_layer][-1].view(activations[selected_layer][-1].shape[0], -1), 2, "PCA")
                fig, ax = plt.subplots()
                sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], ax=ax)
                st.pyplot(fig)

        with col2:
            if "UMAP" in analysis_types:
                st.subheader("UMAP")
                umap_data = dim_reduction.plot_umap(activations[selected_layer][-1].view(activations[selected_layer][-1].shape[0], -1), selected_layer, 5, 2)
                fig, ax = plt.subplots()
                sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
