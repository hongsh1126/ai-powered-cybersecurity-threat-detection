import numpy as np

from security_ml.data import generate_security_events
from security_ml.model import SecurityThreatDetector
from security_ml.schema import FEATURES, TARGET


def test_generator_is_deterministic():
    first = generate_security_events(rows=200, seed=7)
    second = generate_security_events(rows=200, seed=7)
    assert first.equals(second)
    assert 0 < first[TARGET].mean() < 1


def test_end_to_end_scoring():
    frame = generate_security_events(rows=700, seed=11)
    model = SecurityThreatDetector().fit(frame[FEATURES], frame[TARGET])
    scored = model.score(frame.head(25))

    assert list(scored.columns) == ["threat_probability", "anomaly_score", "risk_score", "alert"]
    assert np.all(scored["threat_probability"].between(0, 1))
    assert np.all(scored["anomaly_score"].between(0, 1))
    assert np.all(scored["risk_score"].between(0, 100))


def test_missing_feature_fails_fast():
    frame = generate_security_events(rows=200)
    model = SecurityThreatDetector()
    try:
        model.fit(frame[FEATURES].drop(columns=["packets"]), frame[TARGET])
    except ValueError as exc:
        assert "packets" in str(exc)
    else:
        raise AssertionError("Missing feature should raise ValueError")
