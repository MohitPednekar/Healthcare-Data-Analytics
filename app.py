
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
ANOMALY_PATH = BASE_DIR / "models" / "anomaly_detector.pkl"

# ==================================================
# 3. LOAD DATA AND MODELS
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    anomaly_model = joblib.load(ANOMALY_PATH)

    return model, preprocessor, anomaly_model


try:
    df = load_data()
    model, preprocessor, anomaly_model = load_models()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

MISSING_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🏥 Healthcare Analytics")
st.sidebar.caption("Machine Learning & Data Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard",
        "📈 Model Performance",
        "🧪 Prediction",
        "🔍 Anomaly Detection"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Project: Healthcare Data Analytics")
st.sidebar.caption("Models: Random Forest & Isolation Forest")

# ==================================================
# 5. DASHBOARD
# ==================================================

if page == "📊 Dashboard":

    st.title("Healthcare Analytics Dashboard")

    st.write(
        "Explore the dataset, understand important features "
        "and analyse diabetes outcome patterns."
    )

    st.divider()

    # KPI CARDS

    total_records = len(df)
    total_features = len(FEATURES)
    positive = int(df["Outcome"].sum())
    negative = total_records - positive

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Total Features", total_features)
    col3.metric("Positive Outcomes", positive)
    col4.metric("Negative Outcomes", negative)

    st.divider()

    # DATA FILTER

    st.subheader("Explore Dataset")

    outcome_filter = st.selectbox(
        "Filter by Outcome",
        ["All", "Positive", "Negative"]
    )

    filtered_df = df.copy()

    if outcome_filter == "Positive":
        filtered_df = filtered_df[
            filtered_df["Outcome"] == 1
        ]

    elif outcome_filter == "Negative":
        filtered_df = filtered_df[
            filtered_df["Outcome"] == 0
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=300
    )

    st.caption(f"Showing {len(filtered_df)} records")

    st.divider()

    # INTERACTIVE CHARTS

    left, right = st.columns(2)

    with left:

        st.subheader("Outcome Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.countplot(
            data=df,
            x="Outcome",
            hue="Outcome",
            palette="Set2",
            legend=False,
            ax=ax
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)
        plt.close(fig)

    # INTERACTIVE FEATURE ANALYSIS

    with right:

        st.subheader("Interactive Feature Analysis")

        selected_feature = st.selectbox(
            "Select a feature to analyse",
            FEATURES,
            key="dashboard_feature"
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=df,
            x=selected_feature,
            hue="Outcome",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_title(f"{selected_feature} Distribution")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # CORRELATION HEATMAP

    st.subheader("Feature Correlation Analysis")

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 6. MODEL PERFORMANCE
# ==================================================

elif page == "📈 Model Performance":

    st.title("Machine Learning Model Evaluation")

    st.write(
        "Comparison of Logistic Regression and Random Forest "
        "using cross-validation and test-set evaluation."
    )

    st.divider()

    # CROSS VALIDATION RESULTS

    cv_results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ],
        "Logistic Regression": [
            0.772, 0.726, 0.571, 0.636, 0.837
        ],
        "Random Forest": [
            0.760, 0.679, 0.597, 0.635, 0.820
        ]
    })

    st.subheader("5-Fold Cross-Validation")

    st.dataframe(
        cv_results.set_index("Metric").style.format("{:.3f}"),
        use_container_width=True
    )

    st.bar_chart(
        cv_results.set_index("Metric")
    )

    st.info(
        "These are average cross-validation results across five folds. "
        "They are different from the separate test-set results."
    )

    st.divider()

    # TEST SET METRICS

    st.subheader("Test Set Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Logistic Regression Accuracy", "70.8%")
        st.metric("Logistic Regression ROC-AUC", "0.813")

    with col2:
        st.metric("Random Forest Accuracy", "77.3%")
        st.metric("Random Forest ROC-AUC", "0.818")

    st.divider()

    # CONFUSION MATRICES

    st.subheader("Confusion Matrix Comparison")

    left, right = st.columns(2)

    with left:

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            [[82, 18], [27, 27]],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax
        )

        ax.set_title("Logistic Regression")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            [[86, 14], [21, 33]],
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax
        )

        ax.set_title("Random Forest")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # FEATURE IMPORTANCE

    st.subheader("Random Forest Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": [
            "Glucose",
            "BMI",
            "Age",
            "DiabetesPedigreeFunction",
            "Pregnancies",
            "Insulin",
            "BloodPressure",
            "SkinThickness"
        ],
        "Importance": [
            0.0923,
            0.0478,
            0.0174,
            0.0128,
            0.0126,
            0.0051,
            0.0010,
            -0.0005
        ]
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean decrease in ROC-AUC")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 7. PREDICTION
# ==================================================

elif page == "🧪 Prediction":

    st.title("Diabetes Prediction Demonstration")

    st.warning(
        "Educational demonstration only. "
        "This model is not intended for medical diagnosis."
    )

    st.write(
        "Enter the eight dataset features to see the "
        "Random Forest model's output."
    )

    st.divider()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies", 0, 20, 2
            )

            glucose = st.number_input(
                "Glucose", 0, 300, 120
            )

            bp = st.number_input(
                "Blood Pressure", 0, 200, 70
            )

            skin = st.number_input(
                "Skin Thickness", 0, 100, 25
            )

        with col2:

            insulin = st.number_input(
                "Insulin", 0, 900, 80
            )

            bmi = st.number_input(
                "BMI", 0.0, 80.0, 28.5
            )

            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                0.0, 3.0, 0.5
            )

            age = st.number_input(
                "Age", 1, 120, 35
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }], columns=FEATURES)

        input_data[MISSING_ZERO_COLUMNS] = (
            input_data[MISSING_ZERO_COLUMNS].replace(0, np.nan)
        )

        processed = preprocessor.transform(input_data)

        prediction = model.predict(processed)[0]

        probability = model.predict_proba(processed)[0][1]

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Positive-Class Probability",
                f"{probability:.2%}"
            )

        with col2:
            st.metric(
                "Predicted Class",
                "Positive" if prediction == 1 else "Negative"
            )

        st.progress(float(probability))

        if prediction == 1:
            st.warning("The model predicts a positive outcome.")
        else:
            st.success("The model predicts a negative outcome.")

        st.caption(
            "This is a model-generated estimate, not a clinically "
            "validated probability or diagnosis."
        )


# ==================================================
# 8. ANOMALY DETECTION
# ==================================================

elif page == "🔍 Anomaly Detection":

    st.title("Anomaly Detection Analysis")

    st.write(
        "Identify statistically unusual records using "
        "the Isolation Forest algorithm."
    )

    st.info(
        "An anomaly is a statistically unusual record. "
        "It does not automatically indicate a medical problem."
    )

    st.divider()

    features = df[FEATURES].copy()

    features[MISSING_ZERO_COLUMNS] = (
        features[MISSING_ZERO_COLUMNS].replace(0, np.nan)
    )

    processed = preprocessor.transform(features)

    predictions = anomaly_model.predict(processed)

    scores = anomaly_model.decision_function(processed)

    report = df.copy()

    report["Anomaly Score"] = scores

    report["Anomaly Status"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    # KPI CARDS

    normal_count = int((predictions == 1).sum())
    anomaly_count = int((predictions == -1).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(report))
    col2.metric("Normal Records", normal_count)
    col3.metric("Anomalous Records", anomaly_count)

    anomaly_percentage = anomaly_count / len(report) * 100

    st.metric(
        "Anomaly Percentage",
        f"{anomaly_percentage:.2f}%"
    )

    st.divider()

    # FILTER

    st.subheader("Filter Anomaly Records")

    status_filter = st.selectbox(
        "Select Record Type",
        ["All", "Normal", "Anomaly"]
    )

    filtered_report = report.copy()

    if status_filter != "All":

        filtered_report = filtered_report[
            filtered_report["Anomaly Status"] == status_filter
        ]

    st.dataframe(
        filtered_report,
        use_container_width=True,
        height=350
    )

    st.caption(f"Showing {len(filtered_report)} records")

    st.divider()

    # ANOMALY VISUALIZATION

    st.subheader("Anomaly Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=report,
        x="Glucose",
        y="BMI",
        hue="Anomaly Status",
        style="Anomaly Status",
        s=70,
        ax=ax
    )

    ax.set_title("Anomaly Detection: Glucose vs BMI")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("BMI")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # DOWNLOAD REPORT

    st.subheader("Export Anomaly Report")

    csv_data = filtered_report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇ Download CSV Report",
        data=csv_data,
        file_name="anomaly_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==================================================
# 9. FOOTER
# ==================================================

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
