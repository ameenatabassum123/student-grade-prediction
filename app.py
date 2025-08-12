import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("your_dataset.csv")  # Replace with your CSV file name

# Show dataset preview
st.write("### Dataset Preview")
st.dataframe(df.head())

# Show all column names
st.write("Dataset Columns:", df.columns.tolist())

# Normalize column names for matching
cols = {col.strip().lower(): col for col in df.columns}

# Possible column name variations
internal_raw = cols.get("internal marks")
preboard_raw = cols.get("preboard marks")
internal_std_col = cols.get("internal marks (standardized)")
preboard_std_col = cols.get("preboard marks (standardized)")
grade_col = cols.get("predicted grade") or cols.get("grade")

if not grade_col:
    st.error("Could not find 'Predicted Grade' or 'Grade' column in the dataset.")
    st.stop()

# ---------------------------
# Determine if we have raw or standardized data
# ---------------------------
if internal_raw and preboard_raw:
    st.write("Raw marks detected — standardizing now.")

    internal_mean = df[internal_raw].mean()
    internal_std = df[internal_raw].std()

    preboard_mean = df[preboard_raw].mean()
    preboard_std = df[preboard_raw].std()

    df["Internal_Standardized"] = (df[internal_raw] - internal_mean) / internal_std
    df["Preboard_Standardized"] = (df[preboard_raw] - preboard_mean) / preboard_std

    X = df[["Internal_Standardized", "Preboard_Standardized"]]
else:
    st.write("Standardized marks already in dataset — skipping standardization.")

    if not internal_std_col or not preboard_std_col:
        st.error("Dataset does not contain the required standardized columns.")
        st.stop()

    X = df[[internal_std_col, preboard_std_col]]

# ---------------------------
# Visualization
# ---------------------------
st.write("### Marks Distribution")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

if internal_raw and preboard_raw:
    ax[0].hist(df[internal_raw], bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title("Internal Marks")
    ax[1].hist(df[preboard_raw], bins=10, color='lightgreen', edgecolor='black')
    ax[1].set_title("Preboard Marks")
else:
    ax[0].hist(df[internal_std_col], bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title("Internal Marks (Standardized)")
    ax[1].hist(df[preboard_std_col], bins=10, color='lightgreen', edgecolor='black')
    ax[1].set_title("Preboard Marks (Standardized)")

st.pyplot(fig)

# ---------------------------
# Model Training
# ---------------------------
y = df[grade_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ---------------------------
# Prediction Section
# ---------------------------
st.write("## Predict Student Grade")

if internal_raw and preboard_raw:
    internal_raw_val = st.number_input("Internal Marks (0 - 20)", min_value=0.0, max_value=20.0, step=0.5)
    preboard_raw_val = st.number_input("Preboard Marks (0 - 60)", min_value=0.0, max_value=60.0, step=0.5)

    internal_std_val = (internal_raw_val - internal_mean) / internal_std
    preboard_std_val = (preboard_raw_val - preboard_mean) / preboard_std

    st.write(f"Standardized Internal Marks: {internal_std_val:.2f}")
    st.write(f"Standardized Preboard Marks: {preboard_std_val:.2f}")

    if st.button("Predict Grade"):
        prediction = model.predict([[internal_std_val, preboard_std_val]])
        st.success(f"Predicted Grade: {prediction[0]}")
else:
    internal_std_val = st.number_input("Internal Marks (Standardized)", step=0.1)
    preboard_std_val = st.number_input("Preboard Marks (Standardized)", step=0.1)

    if st.button("Predict Grade"):
        prediction = model.predict([[internal_std_val, preboard_std_val]])
        st.success(f"Predicted Grade: {prediction[0]}")
