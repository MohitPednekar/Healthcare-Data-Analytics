# Healthcare Data Analytics & Anomaly Detection System

## Overview
An end-to-end healthcare data analytics project built using Python and Machine Learning. The system performs data cleaning, exploratory data analysis, diabetes outcome prediction, and anomaly detection. It also provides an interactive dashboard developed using Streamlit.

This project is intended for educational and analytical purposes and is not a medical diagnostic tool.

## Features
- Data cleaning and preprocessing using Pandas and NumPy.
- Exploratory Data Analysis (EDA) with visualisations.
- Diabetes outcome classification using Logistic Regression and Random Forest.
- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation.
- Anomaly detection using Isolation Forest.
- Feature importance analysis using permutation importance.
- Interactive Streamlit dashboard.
- Downloadable anomaly detection report.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure
```text
Healthcare-Data-Analytics/
│
├── data/
│   ├── diabetes.csv
│   └── anomaly_report.csv
│
├── notebooks/
│   └── 01_Data_Exploration.ipynb
│
├── models/
│   ├── random_forest.pkl
│   ├── preprocessor.pkl
│   └── anomaly_detector.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Algorithms

### 1. Logistic Regression
Used as a baseline classification algorithm for predicting diabetes outcomes.

### 2. Random Forest Classifier
Used for classification and comparison with Logistic Regression.

### 3. Isolation Forest
An unsupervised machine learning algorithm used to identify statistically unusual records.

## Model Evaluation
The classification models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold Stratified Cross-Validation
- Confusion Matrix

## Installation and Execution

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder
```bash
cd Healthcare-Data-Analytics
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Limitations
- The project uses a relatively small dataset.
- Dataset patterns may not generalise to other populations.
- Anomaly detection identifies statistically unusual records, not necessarily incorrect or clinically significant records.
- Predictions are for educational demonstration only and should not be used for medical decisions.

## Future Improvements
- Test the models on additional datasets.
- Improve model performance and validation.
- Add more interactive visualisations.
- Deploy the application online.

## Author
Developed as a Machine Learning and Data Analytics project.