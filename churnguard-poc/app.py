# Basic Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Title Page
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

st.title("🔥 ChurnGuard AI – Dual Mode Churn Prediction (ML + Sentiment)")
st.write("""
Upload:
- **Customer Data (numerical/categorical)** → ML-based churn prediction  
- **Customer Feedback (text)** → Sentiment-based churn prediction  

The app automatically routes to the correct analysis mode.
""")

st.header("📌 Step 1 — Upload Your Input Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Customer Data (Numerical & Categorical)")
    num_file = st.file_uploader("Upload Customer CSV", type=["csv"], key="numerical")

with col2:
    st.subheader("📝 Customer Feedback (Sentiment)")
    sent_file = st.file_uploader("Upload Feedback CSV", type=["csv"], key="sentiment")

# Flags
numerical_uploaded = num_file is not None
sentiment_uploaded = sent_file is not None

# SENTIMENT ANALYSIS
if sentiment_uploaded and not numerical_uploaded:

    st.header("🧠 Sentiment-Based Churn Prediction")

    fb = pd.read_csv(sent_file)

    if "feedback" not in fb.columns:
        st.error("The Feedback CSV must contain a **feedback** column.")
        st.stop()

    # Make customer_id optional
    if "customer_id" not in fb.columns:
        fb["customer_id"] = fb.index.astype(str)

    # Convert to string
    fb["customer_id"] = fb["customer_id"].astype(str)

    # Sentiment Scores
    fb["sentiment_score"] = fb["feedback"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    fb["sentiment_label"] = fb["sentiment_score"].apply(
        lambda s: "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
    )

    fb["sentiment_churn_risk"] = (1 - fb["sentiment_score"]).clip(0, 1)

    st.subheader("📊 Sentiment-Based Churn Scores")
    st.write(fb.head(20))

    st.download_button(
        "Download Sentiment Churn CSV",
        fb.to_csv(index=False),
        "sentiment_churn_results.csv",
        "text/csv"
    )

    st.stop()

# Predictive Machine Learning Modelling
if numerical_uploaded and not sentiment_uploaded:

    st.header("🧠 ML-Based Churn Prediction")

    df = pd.read_csv(num_file)
    st.subheader("Customer Data Preview")
    st.write(df.head())

    # Ensure customer_id exists
    if "customer_id" not in df.columns:
        df["customer_id"] = df.index.astype(str)

    df["customer_id"] = df["customer_id"].astype(str)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.header("1. Select or Create Churn Column")

    target_choice = st.sidebar.selectbox(
        "Select churn target column",
        ["--auto-create--"] + list(df.columns)
    )

    if target_choice == "--auto-create--":
        st.warning("Auto-creating churn column...")

        if len(numeric_cols) > 0:
            metric = numeric_cols[0]
            cutoff = df[metric].quantile(0.30)
            df["churn"] = (df[metric] <= cutoff).astype(int)
        else:
            df["churn"] = np.random.randint(0, 2, len(df))

        target = "churn"
    else:
        target = target_choice

    if df[target].nunique() < 2:
        df[target] = np.random.randint(0, 2, len(df))

    st.sidebar.header("2. Feature Selection")

    available_features = [c for c in df.columns if c not in ["customer_id", target]]

    features = st.sidebar.multiselect(
        "Choose features for training",
        available_features,
        default=available_features[:5]
    )

    if not features:
        features = numeric_cols.copy()

    X = df[features].copy()
    y = df[target].astype(int)

    # Preprocess data
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if num_cols:
        imp = SimpleImputer(strategy="median")
        X[num_cols] = imp.fit_transform(X[num_cols])

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Balance if needed
    if df[target].value_counts().min() < 0.1 * len(df):
        X["__target__"] = y
        maj = X[X["__target__"] == 0]
        mino = X[X["__target__"] == 1]
        min_up = resample(mino, replace=True, n_samples=len(maj), random_state=42)
        X = pd.concat([maj, min_up])
        y = X["__target__"]
        X = X.drop(columns=["__target__"])

    st.sidebar.header("3. Choose ML Model")
    model_name = st.sidebar.selectbox(
        "Select model",
        ["Logistic Regression", "Random Forest", "Decision Tree",
         "Gradient Boosting", "KNN", "SVM", "Naive Bayes"]
    )

    def get_model(name):
        return {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=200),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB(),
        }[name]

    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

    if st.button("Train Model"):
        model = get_model(model_name)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Predictive step
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            proba = np.zeros(len(preds))

        st.subheader("📊 Model Evaluation")
        st.text(classification_report(y_test, preds))

        try:
            st.write("ROC-AUC:", roc_auc_score(y_test, proba))
        except:
            st.write("ROC-AUC unavailable")

        # Full dataset churn score
        try:
            df["ml_churn_score"] = model.predict_proba(X)[:, 1]
        except:
            df["ml_churn_score"] = 0

        st.subheader("🔥 ML Churn Results")
        st.write(df.sort_values("ml_churn_score", ascending=False).head(20))

        st.download_button(
            "Download ML Churn CSV",
            df.to_csv(index=False),
            "ml_churn_results.csv",
            "text/csv"
        )

        # Save model
        joblib.dump(model, "model.pkl")
        with open("model.pkl", "rb") as f:
            st.download_button("Download model.pkl", f, "model.pkl")

    st.stop()

# Sentiment Analysis + Predictive ML Modelling
if numerical_uploaded and sentiment_uploaded:

    st.header("🔥 Combined ML + Sentiment Churn Prediction")

    df = pd.read_csv(num_file)
    fb = pd.read_csv(sent_file)

    if "customer_id" not in df.columns:
        df["customer_id"] = df.index.astype(str)
    if "customer_id" not in fb.columns:
        fb["customer_id"] = fb.index.astype(str)

    df["customer_id"] = df["customer_id"].astype(str)
    fb["customer_id"] = fb["customer_id"].astype(str)

    if "feedback" not in fb.columns:
        st.error("Feedback CSV must contain a 'feedback' column.")
        st.stop()

    fb["sentiment_score"] = fb["feedback"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    fb["sentiment_label"] = fb["sentiment_score"].apply(
        lambda s: "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
    )

    # ---------------- Predictive ML Model ----------------
    if "churn" not in df.columns:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 0:
            metric = numeric_cols[0]
            cutoff = df[metric].quantile(0.30)
            df["churn"] = (df[metric] <= cutoff).astype(int)
        else:
            df["churn"] = np.random.randint(0, 2, len(df))

    y = df["churn"]
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if num_cols:
        imp = SimpleImputer(strategy="median")
        X[num_cols] = imp.fit_transform(X[num_cols])

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    df["ml_churn_score"] = model.predict_proba(X)[:, 1]

    # ---------------- MERGE ----------------
    merged = df.merge(fb, on="customer_id", how="left")

    # 70% ML + 30% Sentiment for final scoring
    merged["final_churn_score"] = (
        0.7 * merged["ml_churn_score"] +
        0.3 * (1 - merged["sentiment_score"].fillna(0))
    ).clip(0, 1)

    st.subheader("🔥 FINAL Combined Churn Ranking")
    st.write(merged.sort_values("final_churn_score", ascending=False).head(20))

    st.download_button(
        "Download Final Combined CSV",
        merged.to_csv(index=False),
        "combined_churn_results.csv",
        "text/csv"
    )
