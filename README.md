# Student Grade Prediction App

A Streamlit web application that predicts student grades using a K-Nearest Neighbors (KNN) machine learning model.

## Features

- **Grade Prediction**: Input standardized internal and preboard marks to predict student grades
- **Data Visualizations**: 
  - Grade distribution chart
  - Scatter plot of internal vs preboard marks
  - Correlation heatmap
- **Interactive Interface**: User-friendly Streamlit interface

## Project Structure

- `app.py` - Main Streamlit application
- `EDA_Formatted_Data.csv` - Training dataset with standardized marks
- `knn_model.pkl` - Trained KNN model
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features**: Internal Marks (Standardized), Preboard Marks (Standardized)
- **Target**: Predicted Grade (0-10 scale)
- 

## My Role
I contributed to:
- *Exploratory Data Analysis (EDA)* to uncover trends and correlations in student performance data.
- *Data Visualization* using Python (Matplotlib, Seaborn, Plotly) to present insights in a clear and interactive format.
- *UI/UX Design & App Development* using Streamlit to build an intuitive web interface where users can:
  - Upload datasets
  - Explore interactive visualizations
  - View prediction results in real-time

## Tech Stack
- *Languages:* Python
- *Libraries:* Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn
- *Framework:* Streamlit
- *Tools:* Jupyter Notebook, Git, GitHub

## Data

The app uses a cleaned dataset with standardized marks for internal assessments and preboard examinations to predict final grades on a 0-10 scale.
