# services/cluster_engine.py
"""
Clustering engine:
- train_and_save_model(students_list, ...)
- cluster_students(student_list) using latest saved artifacts
- load_model_and_scaler() helper
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Where to save artifacts (make sure this folder is writable by your app)
MODEL_PATH = 'clustering/models/kmeans_model.pkl'
SCALER_PATH = 'clustering/models/kmeans_scaler.pkl'
PCA_PATH    = 'clustering/models/kmeans_pca.pkl'

# IMPORTANT: This must match your preprocessing output exactly.
# If you add/remove features in your pipeline, update this list accordingly.
FEATURES: List[str] = [
    'Academic Performance Change',
    'Workload Rating',
    'Learning_Visual',
    'Learning_Auditory',
    'Learning_Reading/Writing',
    'Learning_Kinesthetic',
    'Help Seeking',
    'Personality',
    'Hobby Count',
    'Financial Status',
    'Birth Order',
    'Has External Responsibilities',
    'Average',
    'Marital_Separated',
    'Marital_Together',
]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_model_and_scaler() -> Tuple[KMeans, StandardScaler, PCA]:
    """Load the latest saved KMeans, StandardScaler, and PCA artifacts."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH)):
        raise FileNotFoundError(
            "Clustering artifacts not found. Train the model first (e.g., via CSV upload)."
        )
    model: KMeans = joblib.load(MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    pca: PCA = joblib.load(PCA_PATH)
    return model, scaler, pca


def _df_from_students(students_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from a list of student dicts, safely coercing to numeric."""
    df = pd.DataFrame(students_list)
    # Keep only known features; coerce to numeric; fill missing with 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df


def train_and_save_model(
    students_list: List[Dict[str, Any]],
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, StandardScaler, PCA, List[Dict[str, Any]], List[float]]:
    """
    Fit a new StandardScaler → PCA → KMeans on the given batch, save artifacts,
    and return the clustered batch.

    Returns:
        (model, scaler, pca, clustered_records, pca_explained_variance_ratio)
    """
    df = _df_from_students(students_list)
    X = df[FEATURES].to_numpy(dtype=float)

    # Scale
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # PCA (cap components to min(11, num_features) for stable downstream usage)
    n_components = min(11, Xs.shape[1])
    pca = PCA(n_components=n_components).fit(Xs)
    Xp = pca.transform(Xs)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(Xp)
    labels = kmeans.predict(Xp)

    # Persist (overwrite artifacts every time we retrain)
    _ensure_dir(MODEL_PATH)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump(pca,     PCA_PATH)

    # Attach cluster labels
    df = df.copy()
    df['cluster'] = labels
    clustered_records = df.to_dict(orient='records')
    return kmeans, scaler, pca, clustered_records, pca.explained_variance_ratio_.tolist()


def cluster_students(student_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict clusters for a batch using the latest saved artifacts.
    """
    df = _df_from_students(student_list)
    model, scaler, pca = load_model_and_scaler()

    X = df[FEATURES].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)

    df = df.copy()
    df['cluster'] = model.predict(Xp)
    return df.to_dict(orient='records')
