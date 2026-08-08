"""Score security events using a trained artifact."""

from __future__ import annotations

import argparse

import pandas as pd

from security_ml.model import SecurityThreatDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/security_model.joblib")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(args.data).head(args.limit)
    detector = SecurityThreatDetector.load(args.model)
    scores = detector.score(frame).sort_values("risk_score", ascending=False)
    print(scores.to_string(index=False, formatters={"risk_score": "{:.1f}".format}))


if __name__ == "__main__":
    main()
