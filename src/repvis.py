import streamlit as st

from app import main

def launch(model, features, labels):
    st.session_state.model = model
    st.session_state.features = features
    st.session_state.labels = labels
    main()
