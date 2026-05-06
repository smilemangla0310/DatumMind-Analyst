"""
ML utility functions for scikit-learn models.
Available to LLM-generated code via the sandboxed executor.
"""

import pandas as pd
import numpy as np
from typing import Any


def prepare_ml_data(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Prepare data for ML: handle encoding, scaling, and train/test split.
    Returns a dict with X_train, X_test, y_train, y_test, feature_names, scaler.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df_clean = df.dropna(subset=[target]).copy()

    if features is None:
        features = [c for c in df_clean.columns if c != target]

    # Encode categorical features
    label_encoders = {}
    for col in features:
        if df_clean[col].dtype == "object":
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le

    # Drop remaining non-numeric columns and rows with NaN
    X = df_clean[features].select_dtypes(include=[np.number])
    valid_features = X.columns.tolist()
    X = X.fillna(X.median())
    y = df_clean[target]

    # Encode target if categorical
    target_encoder = None
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target)

    # Scale features
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=valid_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=valid_features, index=X_test.index)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": valid_features,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
    }


def evaluate_model(model, X_test, y_test, task_type: str = "regression") -> dict[str, Any]:
    """
    Evaluate a trained model and return metrics.
    task_type: 'regression', 'classification', 'clustering', or 'anomaly'
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        silhouette_score,
    )

    metrics: dict[str, Any] = {"task_type": task_type}

    if task_type == "regression":
        y_pred = model.predict(X_test)
        metrics["r2_score"] = round(r2_score(y_test, y_pred), 4)
        metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    elif task_type == "classification":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        avg = "weighted" if len(set(y_test)) > 2 else "binary"
        metrics["precision"] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        metrics["recall"] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
        metrics["f1_score"] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

    elif task_type == "clustering":
        labels = model.labels_
        if len(set(labels)) > 1:
            metrics["silhouette_score"] = round(silhouette_score(X_test, labels), 4)
        metrics["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics["cluster_sizes"] = dict(pd.Series(labels).value_counts().to_dict())

    elif task_type == "anomaly":
        y_pred = model.predict(X_test)
        n_anomalies = int((y_pred == -1).sum())
        metrics["n_anomalies"] = n_anomalies
        metrics["anomaly_pct"] = round(n_anomalies / len(y_pred) * 100, 2)

    return metrics


def format_ml_report(model, metrics: dict, features: list[str]) -> str:
    """Generate a human-readable ML model report."""
    lines = []
    model_name = type(model).__name__
    task = metrics.get("task_type", "unknown")

    lines.append(f"## Model: {model_name}")
    lines.append(f"**Task:** {task.title()}")
    lines.append(f"**Features used ({len(features)}):** {', '.join(features)}")
    lines.append("")

    lines.append("### Performance Metrics")
    for key, value in metrics.items():
        if key == "task_type":
            continue
        if isinstance(value, dict):
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        else:
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    # Feature importance if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        lines.append("")
        lines.append("### Feature Importance (Top 10)")
        for feat, imp in feature_imp[:10]:
            bar = "█" * int(imp * 40)
            lines.append(f"- {feat}: {imp:.4f} {bar}")

    elif hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        if len(coefs) == len(features):
            feature_coef = sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True)
            lines.append("")
            lines.append("### Coefficients (Top 10)")
            for feat, coef in feature_coef[:10]:
                lines.append(f"- {feat}: {coef:.4f}")

    return "\n".join(lines)
