import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODELS_DIR=PROJECT_ROOT / "models"
RESULTS_DIR=PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

X,y=load_iris(return_X_y=True)
X=pd.DataFrame(X, columns=load_iris().feature_names)
y=pd.Series(y)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)


models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True)
}

results=[]

for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    

    metrics={
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro")
    }
    
    results.append([name] + list(metrics.values()))
    

    joblib.dump(model,MODELS_DIR / f"{name}.joblib")


df_results=pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
df_results.to_csv(RESULTS_DIR / "model_comparison.csv",index=False)

print("Training complete! Models and results saved successfully.")
print(df_results)
