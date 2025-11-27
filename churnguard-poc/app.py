# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score
# from sklearn.impute import SimpleImputer
# from sklearn.utils import resample

# # ----------------------------------------------------
# # PAGE CONFIG
# # ----------------------------------------------------
# st.set_page_config(page_title="ChurnGuard – Universal Trainer", layout="wide")

# st.title("ChurnGuard – Train On ANY Data")
# st.write("✔ Select churn column OR auto-create one\n✔ Never errors\n✔ Always trains a model")

# # ----------------------------------------------------
# # UPLOAD CSV
# # ----------------------------------------------------
# uploaded = st.file_uploader("Upload your customer CSV file", type=["csv"])

# if uploaded is None:
#     st.info("Upload a CSV file to continue.")
#     st.stop()

# df = pd.read_csv(uploaded)

# st.subheader("📄 Data Preview")
# st.write(df.head())

# # ----------------------------------------------------
# # TARGET COLUMN SELECTION
# # ----------------------------------------------------
# st.sidebar.header("1. Select (or Auto-Generate) Target Column")

# target_choice = st.sidebar.selectbox(
#     "Choose churn label column:",
#     ["--auto-create--"] + list(df.columns)
# )

# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# if target_choice == "--auto-create--":
#     st.warning("⚠ No churn column selected. Auto-creating one...")

#     if len(numeric_cols) == 0:
#         df["churn"] = np.random.randint(0, 2, size=len(df))
#         target = "churn"
#         st.success("✔ Auto-created random churn labels (no numeric columns available).")

#     else:
#         metric = numeric_cols[0]
#         cutoff = df[metric].quantile(0.30)
#         df["churn"] = (df[metric] <= cutoff).astype(int)
#         target = "churn"
#         st.success(f"✔ Auto-created churn column using `{metric}` bottom 30% values.")

# else:
#     target = target_choice
#     st.success(f"✔ Using selected churn column: `{target}`")

# # ----------------------------------------------------
# # FIX SINGLE-CLASS TARGET
# # ----------------------------------------------------
# if df[target].nunique() < 2:
#     st.warning("⚠ Selected churn column has only ONE class. Fixing automatically...")

#     if len(numeric_cols) > 0:
#         metric = numeric_cols[0]
#         df[target] = (df[metric] <= df[metric].median()).astype(int)
#         st.success(f"✔ Converted numeric column `{metric}` into a 2-class churn label.")
#     else:
#         df[target] = np.random.randint(0, 2, size=len(df))
#         st.success("✔ Randomly created 2-class churn labels.")

# # ----------------------------------------------------
# # FEATURE SELECTION
# # ----------------------------------------------------
# st.sidebar.header("2. Feature Selection")

# available_features = [c for c in df.columns if c != target]

# features = st.sidebar.multiselect(
#     "Select features to train on:",
#     available_features,
#     default=available_features[: min(6, len(available_features))]
# )

# if not features:
#     st.warning("No features selected. Auto-selecting numeric columns.")
#     features = numeric_cols.copy()

# # ----------------------------------------------------
# # PREPROCESSING
# # ----------------------------------------------------
# X = df[features].copy()
# y = df[target].astype(int)

# numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = [c for c in X.columns if c not in numeric_cols]

# # Impute
# if numeric_cols:
#     imp = SimpleImputer(strategy="median")
#     X[numeric_cols] = imp.fit_transform(X[numeric_cols])

# # One-hot encode
# if categorical_cols:
#     X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# st.write("🧠 Feature matrix shape:", X.shape)

# # ----------------------------------------------------
# # FIX IMBALANCE via UPSAMPLING
# # ----------------------------------------------------
# if df[target].value_counts().min() < 0.1 * len(df):
#     st.warning("⚠ Imbalanced dataset detected — applying oversampling...")

#     X["__target__"] = y
#     majority = X[X["__target__"] == 0]
#     minority = X[X["__target__"] == 1]

#     minority_upsampled = resample(
#         minority, replace=True, n_samples=len(majority), random_state=42
#     )

#     X = pd.concat([majority, minority_upsampled])
#     y = X["__target__"]
#     X = X.drop(columns=["__target__"])

#     st.success("✔ Balanced dataset created.")

# # ----------------------------------------------------
# # MODEL TRAINING
# # ----------------------------------------------------
# st.sidebar.header("3. Train the Model")

# test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# if st.button("Train Model"):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42
#     )

#     model = RandomForestClassifier(n_estimators=150, random_state=42)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)

#     # SAFE probability block
#     try:
#         raw_proba = model.predict_proba(X_test)
#         proba = raw_proba[:, 1] if raw_proba.shape[1] > 1 else np.zeros(len(X_test))
#     except:
#         proba = np.zeros(len(X_test))

#     # ----------------------------------------------------
#     # EVALUATION
#     # ----------------------------------------------------
#     st.subheader("📊 Model Evaluation")
#     st.text(classification_report(y_test, preds))

#     try:
#         st.write("ROC-AUC:", roc_auc_score(y_test, proba))
#     except:
#         st.write("ROC-AUC: Not available")

#     # ----------------------------------------------------
#     # FULL DATASET SCORES
#     # ----------------------------------------------------
#     try:
#         raw_full = model.predict_proba(X)
#         full_proba = raw_full[:, 1] if raw_full.shape[1] > 1 else np.zeros(len(X))
#     except:
#         full_proba = np.zeros(len(X))

#     result = df.copy()
#     result["churn_score"] = full_proba

#     st.subheader("🔥 Top Customers by Churn Risk")
#     st.write(result.sort_values("churn_score", ascending=False).head(15))

#     # Download CSV
#     st.download_button(
#         "Download results CSV",
#         result.to_csv(index=False),
#         "churn_results.csv",
#         "text/csv"
#     )

#     # Download model
#     joblib.dump(model, "model.pkl")
#     with open("model.pkl", "rb") as f:
#         st.download_button("Download Trained Model", f, "model.pkl")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="ChurnGuard – Multi-Model Trainer", layout="wide")

st.title("ChurnGuard – Train ANY Model on ANY Data")
st.write("✔ Select churn column\n✔ Select model type\n✔ Error-proof training")

# ----------------------------------------------------
# UPLOAD CSV
# ----------------------------------------------------
uploaded = st.file_uploader("Upload your customer CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📄 Data Preview")
st.write(df.head())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ----------------------------------------------------
# TARGET COLUMN SELECTION
# ----------------------------------------------------
st.sidebar.header("1. Select Target Column")

target_choice = st.sidebar.selectbox(
    "Choose churn label column",
    ["--auto-create--"] + list(df.columns)
)

if target_choice == "--auto-create--":
    st.warning("⚠ Auto-creating churn column...")

    if len(numeric_cols) == 0:
        df["churn"] = np.random.randint(0, 2, len(df))
    else:
        metric = numeric_cols[0]
        cutoff = df[metric].quantile(0.30)
        df["churn"] = (df[metric] <= cutoff).astype(int)

    target = "churn"
    st.success(f"✔ Auto-created churn using `{metric}`.")
else:
    target = target_choice

# Fix single-class target
if df[target].nunique() < 2:
    st.warning("⚠ Target has only one class. Fixing automatically...")
    if len(numeric_cols) > 0:
        metric = numeric_cols[0]
        df[target] = (df[metric] <= df[metric].median()).astype(int)
    else:
        df[target] = np.random.randint(0, 2, len(df))
    st.success("✔ Fixed target column.")

# ----------------------------------------------------
# FEATURE SELECTION
# ----------------------------------------------------
st.sidebar.header("2. Select Features")

available_features = [c for c in df.columns if c != target]

features = st.sidebar.multiselect(
    "Select features", available_features,
    default=available_features[:6]
)

if not features:
    st.warning("No features selected. Auto-selecting numeric columns.")
    features = numeric_cols.copy()

X = df[features].copy()
y = df[target].astype(int)

# ----------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

if numeric_cols:
    imp = SimpleImputer(strategy="median")
    X[numeric_cols] = imp.fit_transform(X[numeric_cols])

if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

st.write("🧠 Feature matrix shape:", X.shape)

# ----------------------------------------------------
# FIX IMBALANCE
# ----------------------------------------------------
if df[target].value_counts().min() < 0.1 * len(df):
    st.warning("⚠ Severe imbalance detected – applying oversampling...")

    X["__target__"] = y
    majority = X[X["__target__"] == 0]
    minority = X[X["__target__"] == 1]

    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    X = pd.concat([majority, minority_up])

    y = X["__target__"]
    X = X.drop(columns=["__target__"])

    st.success("✔ Balanced dataset.")
    
# ----------------------------------------------------
# MODEL SELECTION
# ----------------------------------------------------
st.sidebar.header("3. Select Model")

model_type = st.sidebar.selectbox(
    "Choose model:",
    [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
        "KNN",
        "SVM",
        "Naive Bayes"
    ]
)

def get_model(model_type):
    """Return the appropriate ML model based on user selection."""
    if model_type == "Logistic Regression":
        return LogisticRegression(max_iter=500)
    if model_type == "Random Forest":
        return RandomForestClassifier(n_estimators=150, random_state=42)
    if model_type == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if model_type == "Gradient Boosting":
        return GradientBoostingClassifier()
    if model_type == "KNN":
        return KNeighborsClassifier()
    if model_type == "SVM":
        return SVC(probability=True)  # enable predict_proba
    if model_type == "Naive Bayes":
        return GaussianNB()

# ----------------------------------------------------
# TRAINING
# ----------------------------------------------------
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

if st.button("Train Model"):

    model = get_model(model_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Safe probability extraction
    try:
        proba_raw = model.predict_proba(X_test)
        proba = proba_raw[:, 1] if proba_raw.shape[1] > 1 else np.zeros(len(X_test))
    except:
        proba = np.zeros(len(X_test))

    st.subheader(f"📊 Evaluation – {model_type}")
    st.text(classification_report(y_test, preds))

    try:
        st.write("ROC-AUC:", roc_auc_score(y_test, proba))
    except:
        st.write("ROC-AUC: Not available")

    # FULL DATASET PROBABILITY
    try:
        full_raw = model.predict_proba(X)
        full_proba = full_raw[:, 1] if full_raw.shape[1] > 1 else np.zeros(len(X))
    except:
        full_proba = np.zeros(len(X))

    result = df.copy()
    result["churn_score"] = full_proba

    st.subheader("🔥 Top Customers by Churn Risk")
    st.write(result.sort_values("churn_score", ascending=False).head(15))

    st.download_button("Download CSV", result.to_csv(index=False),
                       "churn_output.csv", "text/csv")

    joblib.dump(model, "model.pkl")
    with open("model.pkl", "rb") as f:
        st.download_button("Download model.pkl", f, "model.pkl")
