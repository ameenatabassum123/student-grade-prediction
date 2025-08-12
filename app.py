import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
csv_file = "EDA_Formatted_Data.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    st.error(f"❌ {csv_file} not found!")

# Load model
model_file = "knn_model.pkl"
if os.path.exists(model_file):
    knn = joblib.load(model_file)
else:
    st.error(f"❌ {model_file} not found!")

# Load scaler (if available)
scaler_file = "scaler.pkl"
if os.path.exists(scaler_file):
    scaler = joblib.load(scaler_file)
else:
    scaler = None  # Will skip scaling if not found

st.title("Student Grade Prediction App (KNN)")

st.header("Predict Student Grade")
internal_marks = st.number_input("Internal Marks (Standardized)", value=0.0, step=0.01)
preboard_marks = st.number_input("Preboard Marks (Standardized)", value=0.0, step=0.01)

if st.button("Predict Grade"):
    if 'knn' in locals():
        # Prepare input
        input_data = [[internal_marks, preboard_marks]]
        
        # Apply scaling if scaler exists
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        prediction = knn.predict(input_data)
        st.success(f"Predicted Grade: {prediction[0]}")

        # Show data visualizations AFTER prediction
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
