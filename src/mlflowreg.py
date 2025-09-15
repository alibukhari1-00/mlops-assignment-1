import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

# ==========================
# Paths
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "model_comparison.csv"

# ==========================
# Load results CSV
# ==========================
df_results = pd.read_csv(RESULTS_CSV)

# ==========================
# Select best model
# ==========================
best_model_row = df_results.loc[df_results["F1"].idxmax()]
best_model_name = best_model_row["Model"]
print(f"Best model based on F1 score: {best_model_name}")

# ==========================
# Register best model
# ==========================
# MLflow Model Registry name
MODEL_REGISTRY_NAME = "Best_Iris_Model"

# Path to saved model
model_path = MODELS_DIR / f"{best_model_name}.joblib"

# Register model
mlflow.sklearn.log_model(
    sk_model=None,  # None because we log existing model
    artifact_path=best_model_name,  # temporary artifact name
    registered_model_name=MODEL_REGISTRY_NAME
)

# Alternatively, you can use MLflow CLI or UI to register the model manually
print(f"✅ Best model '{best_model_name}' ready for registration in MLflow Model Registry.")
