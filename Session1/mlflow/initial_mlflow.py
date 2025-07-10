"""
Module 1: Modern Industrial AI Workflow - Toy Exercise
Industrial Application of Artificial Intelligence Course

This file contains toy exercises using a synthetic dataset for classification.
Students will work through a complete MLflow workflow for industrial classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# =============================================================================
# EXERCISE 1: MLflow Setup and Toy Dataset Exploration
# =============================================================================

def setup_mlflow_environment(exp_name="toy_classification"):
    """
    Set up MLflow tracking environment for toy classification project.
    """
    print("=== Setting up MLflow Environment ===")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(exp_name)
    mlflow.set_experiment_tag("project", "industrial_ai_course")
    mlflow.set_experiment_tag("domain", "toy_example")
    mlflow.set_experiment_tag("task", "classification")
    return mlflow.get_experiment_by_name(exp_name).experiment_id

def create_toy_dataset(n_samples=500, n_features=5, random_state=42):
    """
    Generate a toy classification dataset.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=3,
        n_redundant=1, n_classes=2, random_state=random_state
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['label'] = y
    print(df.head())
    return df

def explore_toy_dataset(df):
    """
    Print basic dataset statistics and plot class balance.
    """
    print("\n=== Dataset Exploration ===")
    print(f"Shape: {df.shape}")
    print("Class balance:\n", df['label'].value_counts())
    sns.countplot(data=df, x="label")
    plt.title("Class Balance")
    plt.show()

# =============================================================================
# EXERCISE 2: Classification Training with MLflow
# =============================================================================

def train_classifier_with_mlflow(df, model_type="random_forest", registered_model_name=None):
    """
    Train a classifier on the toy dataset and log everything to MLflow.
    Optionally register the model to the MLflow Model Registry.
    """
    print(f"\n=== Training {model_type} ===")
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model selection
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model_type!")

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        mlflow.log_param("model_type", model_type)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1 Score: {f1:.3f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log and (optionally) register model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train_scaled[:2],
            registered_model_name=registered_model_name,  # If None, will just log, not register
        )

        # Log confusion matrix as artifact (image)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        return run.info.run_id, acc


# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

if __name__ == "__main__":
    print("=== Module 1: Toy Classification ===\n")
    setup_mlflow_environment("toy_classification")
    df = create_toy_dataset()
    explore_toy_dataset(df)

    # Train both models, but don't register yet
    run_id_rf, acc_rf = train_classifier_with_mlflow(df, "random_forest")
    run_id_lr, acc_lr = train_classifier_with_mlflow(df, "logistic_regression")

    print("\n=== Toy Classification Complete! ===")
    print(f"🎯 RF Run ID: {run_id_rf} - Accuracy: {acc_rf:.3f}")
    print(f"🎯 LR Run ID: {run_id_lr} - Accuracy: {acc_lr:.3f}")

    # Register the best one!
    if acc_rf >= acc_lr:
        print("\nRegistering the Random Forest as best model.")
        # Re-train and register in one go for the registry (you could optimize to reuse the run if desired)
        train_classifier_with_mlflow(df, "random_forest", registered_model_name="toy-best-classifier")
    else:
        print("\nRegistering the Logistic Regression as best model.")
        train_classifier_with_mlflow(df, "logistic_regression", registered_model_name="toy-best-classifier")

    print("Check MLflow UI to see results and registered model!")
