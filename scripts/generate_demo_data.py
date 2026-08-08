"""Generate safe synthetic telemetry for the demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from security_ml.data import generate_security_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/demo_security_events.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_security_events(rows=args.rows, seed=args.seed)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} events to {output} (attack rate={frame.label.mean():.1%})")


if __name__ == "__main__":
    main()
