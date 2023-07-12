import streamlit as st
st.set_page_config(page_title="Salary Prediction App")

from predict_page import show_predict_page
from explore_page import show_explore_page
from resource_page import resource_page



st.title("Predict Software Engineer Salaries")

predict, explore, resource = st.tabs(["Predict", "Explore", "Resource"])

with predict:
    show_predict_page()
with explore:
    show_explore_page()
with resource:
    resource_page()
