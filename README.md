# AI-Powered Cybersecurity Threat Detection

An end-to-end machine-learning reference project for prioritizing cybersecurity alerts from network-flow telemetry. It combines supervised threat classification with unsupervised anomaly detection, produces analyst-friendly risk scores, and exposes a small CLI for reproducible training and inference.

> Portfolio focus: security telemetry, alert classification, anomaly detection, model evaluation, and explainable ML — the core workflow expected in Security ML/AI engineering roles.

## What this project demonstrates

- Builds a security-event feature pipeline without leaking labels into preprocessing.
- Trains a calibrated Random Forest threat classifier.
- Trains an Isolation Forest alongside it to surface unusual events, including potentially unseen behavior.
- Fuses model signals into a 0–100 risk score for alert triage.
- Reports ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix, and top feature importances.
- Saves one deployable artifact containing preprocessing, both models, and metadata.
- Includes deterministic synthetic telemetry so the repository runs immediately without shipping sensitive logs.

## Architecture

```text
Network / security events
        |
        v
Validation + feature engineering
        |
        +-------------------+
        |                   |
        v                   v
Threat classifier     Isolation Forest
        |                   |
        +---------+---------+
                  v
             Risk fusion
                  |
                  v
        Analyst-ready alerts
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.generate_demo_data --rows 6000
python -m scripts.train --data data/demo_security_events.csv
python -m scripts.predict --data data/demo_security_events.csv --limit 10
pytest -q
```

Training writes `artifacts/security_model.joblib` and `artifacts/metrics.json`. Generated data and model binaries are intentionally ignored by Git.

## Input schema

The baseline uses flow-level fields that are common in IDS / network telemetry exports:

| Field | Type | Meaning |
|---|---|---|
| `duration_ms` | numeric | Flow duration in milliseconds |
| `src_bytes` | numeric | Bytes sent by source |
| `dst_bytes` | numeric | Bytes sent by destination |
| `packets` | numeric | Packet count |
| `failed_logins` | numeric | Failed authentication attempts |
| `connection_rate` | numeric | Recent connection rate |
| `unique_dst_ports` | numeric | Destination-port diversity |
| `protocol` | categorical | `tcp`, `udp`, `icmp`, ... |
| `service` | categorical | Observed application/service |
| `label` | target | `0` benign, `1` malicious |

To use real telemetry, export these fields (or adapt `security_ml/schema.py`) and pass the CSV to `scripts/train.py`.

## Risk scoring

The final score deliberately keeps the two ML signals visible:

```text
risk = 100 * (0.75 * P(malicious) + 0.25 * anomaly_score)
```

The supervised probability carries most weight because it is optimized against known labels. The anomaly component adds a weaker signal for behavior that differs from the training distribution. These weights are a transparent baseline, not a claim of production-optimal calibration.

## Why PR-AUC is included

Security events are typically imbalanced. Accuracy alone can look excellent while missing attacks. The training report therefore emphasizes recall, precision, F1 and PR-AUC alongside ROC-AUC.

## Project structure

```text
security_ml/
  data.py          synthetic security telemetry generator
  model.py         preprocessing, classifier, anomaly detector, risk scoring
  schema.py        validated feature contract
scripts/
  generate_demo_data.py
  train.py
  predict.py
tests/
  test_pipeline.py
```

## Production hardening roadmap

1. Replace synthetic data with approved enterprise telemetry (SIEM, EDR, NetFlow, authentication events).
2. Use time-based splits to measure temporal drift rather than relying only on a random holdout.
3. Add model and feature drift monitoring plus scheduled retraining gates.
4. Calibrate alert thresholds to analyst capacity and business cost, not a fixed 0.5 threshold.
5. Add SHAP explanations and analyst feedback loops for false-positive reduction.
6. Add experiment tracking, model registry, CI security scanning, and containerized inference.

## Responsible use

This repository is defensive. The demo data are synthetic and the project contains no exploit code, credentials, malware, or instructions for unauthorized access. A production deployment should follow the organization's privacy, retention, access-control, and incident-response policies.

## License

MIT — see [LICENSE](LICENSE).
