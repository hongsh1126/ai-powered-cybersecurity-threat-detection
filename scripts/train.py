"""Train and evaluate the security threat detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from security_ml.model import SecurityThreatDetector
from security_ml.schema import FEATURES, TARGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/security_model.joblib")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if TARGET not in frame:
        raise ValueError(f"Training data must contain '{TARGET}'")
    train, test = train_test_split(
        frame, test_size=0.25, random_state=42, stratify=frame[TARGET]
    )

    detector = SecurityThreatDetector().fit(train[FEATURES], train[TARGET])
    scores = detector.score(test[FEATURES])
    predicted = scores["alert"].astype(int)
    metrics = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "test_attack_rate": float(test[TARGET].mean()),
        "roc_auc": float(roc_auc_score(test[TARGET], scores["threat_probability"])),
        "pr_auc": float(average_precision_score(test[TARGET], scores["threat_probability"])),
        "precision": float(precision_score(test[TARGET], predicted, zero_division=0)),
        "recall": float(recall_score(test[TARGET], predicted, zero_division=0)),
        "f1": float(f1_score(test[TARGET], predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(test[TARGET], predicted).tolist(),
        "top_features": detector.feature_importance().to_dict(orient="records"),
    }

    model_path, metrics_path = Path(args.model), Path(args.metrics)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
