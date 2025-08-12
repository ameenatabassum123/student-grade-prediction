import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("your_dataset.csv")  # Replace with your CSV name

# Show dataset preview
st.write("### Dataset Preview")
st.dataframe(df.head())

# Show all column names
st.write("Dataset Columns:", df.columns.tolist())

# Find columns ignoring case and spaces
cols = {col.strip().lower(): col for col in df.columns}

# Match internal marks & preboard marks columns
internal_col = cols.get("internal marks")
preboard_col = cols.get("preboard marks")
grade_col = cols.get("grade")  # Assuming you have Grade column

if not internal_col or not preboard_col or not grade_col:
    st.error("Could not find required columns in the dataset. Please check your CSV.")
    st.stop()

# ---------------------------
# Standardization
# ---------------------------
internal_mean = df[internal_col].mean()
internal_std = df[internal_col].std()

preboard_mean = df[preboard_col].mean()
preboard_std = df[preboard_col].std()

df["Internal_Standardized"] = (df[internal_col] - internal_mean) / internal_std
df["Preboard_Standardized"] = (df[preboard_col] - preboard_mean) / preboard_std

# ---------------------------
# Visualization
# ---------------------------
st.write("### Marks Distribution")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(df[internal_col], bins=10, color='skyblue', edgecolor='black')
ax[0].set_title("Internal Marks")
ax[1].hist(df[preboard_col], bins=10, color='lightgreen', edgecolor='black')
ax[1].set_title("Preboard Marks")
st.pyplot(fig)

# ---------------------------
# Model Training
# ---------------------------
X = df[["Internal_Standardized", "Preboard_Standardized"]]
y = df[grade_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ---------------------------
# Prediction Section
# ---------------------------
st.write("## Predict Student Grade")

# Take raw marks as input
internal_raw = st.number_input("Internal Marks (0 - 20)", min_value=0.0, max_value=20.0, step=0.5)
preboard_raw = st.number_input("Preboard Marks (0 - 60)", min_value=0.0, max_value=60.0, step=0.5)

# Standardize them
internal_std_value = (internal_raw - internal_mean) / internal_std
preboard_std_value = (preboard_raw - preboard_mean) / preboard_std

st.write(f"Standardized Internal Marks: {internal_std_value:.2f}")
st.write(f"Standardized Preboard Marks: {preboard_std_value:.2f}")

if st.button("Predict Grade"):
    prediction = model.predict([[internal_std_value, preboard_std_value]])
    st.success(f"Predicted Grade: {prediction[0]}")
