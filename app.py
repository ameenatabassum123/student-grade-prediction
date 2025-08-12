import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Standardization parameters =====
# Replace with the exact values used in preprocessing
INTERNAL_MEAN = 10    # midpoint of 0–20
INTERNAL_STD = 4      # adjust if your dataset used a different std
PREBOARD_MEAN = 30    # midpoint of 0–60
PREBOARD_STD = 10     # adjust if your dataset used a different std

# ===== Load dataset =====
csv_file = "EDA_Formatted_Data.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.success("✅ Dataset loaded successfully!")
else:
    st.error(f"❌ {csv_file} not found!")

# ===== Load model =====
model_file = "knn_model.pkl"
if os.path.exists(model_file):
    knn = joblib.load(model_file)
    st.success("✅ Model loaded successfully!")
else:
    st.error(f"❌ {model_file} not found!")

st.title("Student Grade Prediction App (KNN)")

# ===== Prediction Section =====
st.header("Predict Student Grade")
internal_marks_raw = st.number_input("Internal Marks (0–20)", 0.0, 20.0, step=0.5)
preboard_marks_raw = st.number_input("Preboard Marks (0–60)", 0.0, 60.0, step=0.5)

if st.button("Predict Grade"):
    if 'knn' in locals():
        # Convert real marks to standardized values for the model
        internal_std = (internal_marks_raw - INTERNAL_MEAN) / INTERNAL_STD
        preboard_std = (preboard_marks_raw - PREBOARD_MEAN) / PREBOARD_STD
        
        prediction = knn.predict([[internal_std, preboard_std]])
        st.success(f"Predicted Grade: {prediction[0]}")

# ===== Visualizations =====
if 'df' in locals():
    st.header("Data Visualizations")
    
    st.subheader("Grade Distribution")
    st.bar_chart(df['Predicted Grade'].value_counts())
    
    st.subheader("Internal vs Preboard Marks")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='Internal Marks (Standardized)', 
        y='Preboard Marks (Standardized)', 
        hue='Predicted Grade', 
        data=df, ax=ax1
    )
    st.pyplot(fig1)
    
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
