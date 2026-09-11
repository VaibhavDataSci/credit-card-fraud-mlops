#!/usr/bin/env python3
"""Explicitly approve or reject a pending candidate promotion report."""

import argparse
import getpass
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.approval import decide_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(BASE_DIR / "reports/model/promotion_decision.json"))
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--model-name", default="CreditCardFraudDetector")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    operator = args.operator or getpass.getuser()
    print(f"Candidate report: {args.report}")
    print("Type APPROVE to promote this candidate, or REJECT to retain the current Champion.")
    decision = input("Decision: ").strip().upper()
    if decision not in {"APPROVE", "REJECT"}:
        print("No action taken. Decision must be exactly APPROVE or REJECT.", file=sys.stderr)
        raise SystemExit(2)
    result = decide_candidate(
        args.report,
        args.tracking_uri,
        args.model_name,
        decision,
        operator,
        args.reason,
    )
    print(f"Final result: {result['approval']['promotion_result']}")
    print(f"Champion version after decision: {result['champion_after']}")


if __name__ == "__main__":
    main()
