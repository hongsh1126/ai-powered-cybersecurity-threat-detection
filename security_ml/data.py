"""Deterministic synthetic telemetry for a safe, runnable demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_security_events(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate flow-like events with learnable but noisy attack patterns."""
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)
    protocol = rng.choice(["tcp", "udp", "icmp"], rows, p=[0.72, 0.23, 0.05])
    service = rng.choice(
        ["https", "dns", "ssh", "smtp", "other"], rows, p=[0.48, 0.18, 0.12, 0.07, 0.15]
    )
    duration_ms = rng.lognormal(mean=6.0, sigma=1.0, size=rows)
    src_bytes = rng.lognormal(mean=7.4, sigma=1.15, size=rows)
    dst_bytes = rng.lognormal(mean=7.2, sigma=1.2, size=rows)
    packets = np.maximum(1, rng.poisson(16, size=rows))
    failed_logins = rng.poisson(0.08, size=rows)
    connection_rate = rng.gamma(shape=2.0, scale=2.2, size=rows)
    unique_dst_ports = np.maximum(1, rng.poisson(2.1, size=rows))

    # Hidden risk function creates attacks while retaining overlap/noise.
    raw_risk = (
        1.35 * (failed_logins >= 2)
        + 1.25 * (connection_rate > 8.5)
        + 1.15 * (unique_dst_ports >= 6)
        + 0.85 * (service == "ssh")
        + 0.60 * (protocol == "icmp")
        + 0.55 * (src_bytes > np.quantile(src_bytes, 0.91))
        + rng.normal(0, 0.8, size=rows)
        - 2.15
    )
    attack_probability = 1.0 / (1.0 + np.exp(-raw_risk))
    label = rng.binomial(1, attack_probability)

    return pd.DataFrame(
        {
            "duration_ms": duration_ms.round(2),
            "src_bytes": src_bytes.round(0),
            "dst_bytes": dst_bytes.round(0),
            "packets": packets,
            "failed_logins": failed_logins,
            "connection_rate": connection_rate.round(3),
            "unique_dst_ports": unique_dst_ports,
            "protocol": protocol,
            "service": service,
            "label": label,
        }
    )
