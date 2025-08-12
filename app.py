import streamlit as st
import pandas as pd
import joblib
import os

INTERNAL_MEAN = 20    # midpoint of 0–40
INTERNAL_STD = 4      # adjust if your dataset used a different std
PREBOARD_MEAN = 30    # midpoint of 0–60
PREBOARD_STD = 10     # adjust if your dataset used a different std


csv_file = "EDA_Formatted_Data.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    st.error(f"❌ {csv_file} not found!")


model_file = "knn_model.pkl"
if os.path.exists(model_file):
    knn = joblib.load(model_file)
else:
    st.error(f"❌ {model_file} not found!")

st.title("Student Grade Prediction App (KNN)")


st.header("Predict Student Grade")
internal_marks_raw = st.number_input("Internal Marks (0–40)", 0.0, 40.0, step=0.5)
preboard_marks_raw = st.number_input("Preboard Marks (0–60)", 0.0, 60.0, step=0.5)

if st.button("Predict Grade"):
    if 'knn' in locals():
       
        internal_std = (internal_marks_raw - INTERNAL_MEAN) / INTERNAL_STD
        preboard_std = (preboard_marks_raw - PREBOARD_MEAN) / PREBOARD_STD

        prediction = knn.predict([[internal_std, preboard_std]])
        st.success(f"Predicted Grade: {prediction[0]}")
