import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'clustering/models/kmeans_model.pkl'
SCALER_PATH = 'clustering/models/kmeans_scaler.pkl'

FEATURES = [
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
    'Marital_Together'
]

def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def cluster_students(student_list):
    """student_list is a list of dicts with the same feature keys"""

    df = pd.DataFrame(student_list)

    model, scaler = load_model_and_scaler()
    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    predicted_clusters = model.predict(X_scaled)
    df['cluster'] = predicted_clusters

    return df.to_dict(orient='records')
