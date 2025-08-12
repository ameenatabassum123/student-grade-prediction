import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
model = joblib.load("knn_model.pkl")
internal_mean = 12.0
internal_std = 4.0
preboard_mean = 40.0
preboard_std = 10.0

st.title("Student Grade Prediction App (KNN)")

internal_marks = st.number_input("Internal Marks (Actual)", min_value=0.0, max_value=20.0, step=0.5)
preboard_marks = st.number_input("Preboard Marks (Actual)", min_value=0.0, max_value=60.0, step=0.5)

if st.button("Predict Grade"):

    internal_std_value = (internal_marks - internal_mean) / internal_std
    preboard_std_value = (preboard_marks - preboard_mean) / preboard_std

    input_data = [[internal_std_value, preboard_std_value]]

    prediction = model.predict(input_data)[0]

    st.write(f"*Predicted Grade:* {prediction}")
    st.caption(f"(Standardized values used: Internal={internal_std_value:.2f}, Preboard={preboard_std_value:.2f})")
