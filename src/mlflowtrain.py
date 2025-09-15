import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
X, y = load_iris(return_X_y=True)
X = pd.DataFrame(X, columns=load_iris().feature_names)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True)
}

results = []

mlflow.set_experiment("mlops_assignment_1_experiment")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1": f1_score(y_test, y_pred, average="macro")
        }
        results.append([name] + list(metrics.values()))

    
        mlflow.log_params(model.get_params())

    
        mlflow.log_metrics(metrics)

        
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, name=name)

    
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = RESULTS_DIR / f"{name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(str(cm_path))
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
csv_path = RESULTS_DIR / "model_comparison.csv"
df_results.to_csv(csv_path, index=False)

mlflow.log_artifact(str(csv_path))

print("✅ MLflow training, logging, and artifacts complete!")
print(df_results)
