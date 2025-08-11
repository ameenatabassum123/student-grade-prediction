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

## Deployment to Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account

### Steps
1. **Push to GitHub**: Upload this cleaned project to a GitHub repository
2. **Connect to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Connect your GitHub repo and deploy
4. **Configuration**: 
   - Main file path: `app.py`
   - Python version: 3.8+ (auto-detected)

### What's Ready for Deployment
✅ All unnecessary files removed  
✅ Dependencies properly specified  
✅ Main app functionality preserved  
✅ Data files optimized  
✅ Project structure clean  

## Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features**: Internal Marks (Standardized), Preboard Marks (Standardized)
- **Target**: Predicted Grade (0-10 scale)

## Data

The app uses a cleaned dataset with standardized marks for internal assessments and preboard examinations to predict final grades on a 0-10 scale.