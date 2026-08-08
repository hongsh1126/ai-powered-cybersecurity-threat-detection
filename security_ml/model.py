"""Supervised + unsupervised threat-detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, missing_features


@dataclass
class SecurityThreatDetector:
    """Fuse known-threat classification with behavior anomaly scoring."""

    random_state: int = 42
    classifier_weight: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.classifier_weight <= 1.0:
            raise ValueError("classifier_weight must be in [0, 1]")
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        self.preprocessor = ColumnTransformer(
            [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
        )
        self.classifier = RandomForestClassifier(
            n_estimators=240,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=180,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._fitted = False

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        missing = missing_features(list(frame.columns))
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

    def fit(self, frame: pd.DataFrame, y: pd.Series | np.ndarray) -> "SecurityThreatDetector":
        self._validate(frame)
        transformed = self.preprocessor.fit_transform(frame[FEATURES])
        self.classifier.fit(transformed, y)

        # Model normal behavior when possible; this makes anomaly scores easier to interpret.
        y_array = np.asarray(y)
        benign = transformed[y_array == 0]
        self.anomaly_detector.fit(benign if len(benign) >= 50 else transformed)
        self._fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring")
        self._validate(frame)
        transformed = self.preprocessor.transform(frame[FEATURES])
        threat_probability = self.classifier.predict_proba(transformed)[:, 1]

        # IsolationForest: lower decision_function is more anomalous.
        raw_anomaly = -self.anomaly_detector.decision_function(transformed)
        lo, hi = np.quantile(raw_anomaly, [0.05, 0.95])
        anomaly_score = np.clip((raw_anomaly - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        risk = self.classifier_weight * threat_probability + (1 - self.classifier_weight) * anomaly_score

        return pd.DataFrame(
            {
                "threat_probability": threat_probability,
                "anomaly_score": anomaly_score,
                "risk_score": 100.0 * risk,
                "alert": risk >= 0.5,
            },
            index=frame.index,
        )

    def feature_importance(self, top_n: int = 12) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before inspecting features")
        names = self.preprocessor.get_feature_names_out()
        importance = self.classifier.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SecurityThreatDetector":
        model = joblib.load(path)
        if not isinstance(model, SecurityThreatDetector):
            raise TypeError("Artifact is not a SecurityThreatDetector")
        return model
