import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC              # Import the new classifier!
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from sklearn.preprocessing import StandardScaler # Added for SVC

# --- 1. DVC PULL & ARTIFACT LOADING ---

# Define paths (must match the paths used with 'dvc add')
PROCESSED_DATA_PATH = 'data/processed/iris_v1_processed.csv'
ENCODER_PATH = 'artifacts/label_encoder_master.joblib'
IMPUTER_PATH = 'artifacts/imputer_means_iris_v1.joblib'

print("Loading data and artifacts...")
df = pd.read_csv(PROCESSED_DATA_PATH)
label_encoder = joblib.load(ENCODER_PATH)
imputer_means = joblib.load(IMPUTER_PATH) 

# Assuming your processed data is ready for training (features and target columns defined)
X = df.drop('target', axis=1)
y = df['target']             

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- IMPORTANT: Scale the data for SVC ---
# SVC is sensitive to feature scaling, so we apply StandardScaler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)


# --- 2. MLFLOW EXPERIMENT TRACKING SETUP ---

# Set the MLflow tracking URI (ensure this is correct)
remote_server_uri = "http://35.224.10.105:8100/"
mlflow.set_tracking_uri(remote_server_uri)

MLFLOW_EXPERIMENT_NAME = "Iris_Classifier_Experiments"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Set the name for the NEW model in the Model Registry
REGISTERED_MODEL_NAME_SVC = "Iris_SVC_Classifier"

# --- 3. NEW MLFLOW RUN: Support_Vector_Machine_V1 ---

# Define hyperparameters for SVC
params_svc = {
    "kernel": "rbf",
    "C": 1.0, 
    "gamma": "scale", # Default: 1 / (n_features * X.var())
    "random_state": 42
}

with mlflow.start_run(run_name="Support_Vector_Machine_V1") as run:
    
    # --- TRAINING (using scaled data) ---
    svc = SVC(**params_svc)
    svc.fit(X_train_scaled, y_train)
    
    # --- EVALUATION ---
    y_pred_svc = svc.predict(X_test_scaled)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    
    # --- MLFLOW LOGGING ---
    
    # 1. Log NEW parameters
    mlflow.log_params(params_svc)
    
    # 2. Log NEW metrics
    mlflow.log_metric("test_accuracy", accuracy_svc)
    
    # 3. Log DVC data version (for full reproducibility)
    git_commit_id = os.popen('git rev-parse HEAD').read().strip()
    mlflow.set_tag("git_commit", git_commit_id)
    mlflow.set_tag("data_version", "v1_processed_data") 
    
    # 4. Log the trained model with Signature
    # NOTE: The StandardScaler object is not included in the model here, 
    # for production, you would typically use a scikit-learn Pipeline 
    # to combine scaling and the classifier before logging.
    mlflow.sklearn.log_model(
        sk_model=svc,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME_SVC,
        # Use scaled dataframe for input example for consistency
        input_example=X_train_scaled_df.head(5) 
    )
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Logged Model Name: {REGISTERED_MODEL_NAME_SVC}")
    print(f"Test Accuracy: {accuracy_svc:.4f}")
    
print("\nSVC Model trained, logged, and registered to MLflow successfully!")