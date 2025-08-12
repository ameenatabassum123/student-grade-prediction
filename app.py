import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD DATA & MODEL
# =========================
df = pd.read_csv("EDA_Formatted_Data.csv")  # Replace with your actual CSV file
model = joblib.load("knn_model.pkl")  # Your trained model

# Calculate mean & std from dataset
internal_mean = df["Internal Marks"].mean()
internal_std = df["Internal Marks"].std()
preboard_mean = df["Preboard Marks"].mean()
preboard_std = df["Preboard Marks"].std()

# =========================
# PAGE TITLE
# =========================
st.title("Student Grade Prediction App (KNN)")

# =========================
# DATA VISUALIZATION SECTION
# =========================
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(df["Internal Marks"])
    st.caption("Distribution of Internal Marks")

with col2:
    st.bar_chart(df["Preboard Marks"])
    st.caption("Distribution of Preboard Marks")

# =========================
# PREDICTION SECTION
# =========================
st.subheader("🎯 Predict Student Grade")

internal_marks = st.number_input(
    "Internal Marks (Actual)", min_value=0.0, max_value=20.0, step=0.5
)
preboard_marks = st.number_input(
    "Preboard Marks (Actual)", min_value=0.0, max_value=60.0, step=0.5
)

if st.button("Predict Grade"):
    # Standardize values
    internal_std_value = (internal_marks - internal_mean) / internal_std
    preboard_std_value = (preboard_marks - preboard_mean) / preboard_std

    # Prepare for prediction
    input_data = [[internal_std_value, preboard_std_value]]
    prediction = model.predict(input_data)[0]

    # Show result
    st.success(f"Predicted Grade: {prediction}")
    st.caption(
        f"(Standardized values used: Internal={internal_std_value:.2f}, Preboard={preboard_std_value:.2f})"
    )
