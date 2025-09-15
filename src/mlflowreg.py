import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "model_comparison.csv"


df_results = pd.read_csv(RESULTS_CSV)


best_model_row = df_results.loc[df_results["F1"].idxmax()]
best_model_name = best_model_row["Model"]
print(f"Best model based on F1 score: {best_model_name}")

MODEL_REGISTRY_NAME = "Best_Iris_Model"


model_path = MODELS_DIR / f"{best_model_name}.joblib"

mlflow.sklearn.log_model(
    sk_model=None,  
    artifact_path=best_model_name, 
    registered_model_name=MODEL_REGISTRY_NAME
)

print(f"Best model '{best_model_name}' ready for registration in MLflow Model Registry.")
