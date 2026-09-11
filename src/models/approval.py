"""Human approval gate for validated MLflow promotion reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient


PENDING_APPROVAL = "PENDING_APPROVAL"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
PROMOTED = "PROMOTED"


def load_pending_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("approval_state", PENDING_APPROVAL) != PENDING_APPROVAL:
        raise ValueError("Promotion report is not pending approval")
    if report.get("recommended_decision") != "PROMOTE":
        raise ValueError("Candidate did not pass automated promotion criteria")
    return report


def _current_champion(client: MlflowClient, model_name: str) -> Any:
    return client.get_model_version_by_alias(model_name, "champion")


def decide_candidate(
    report_path: str | Path,
    tracking_uri: str,
    model_name: str,
    decision: str,
    operator: str,
    reason: str = "",
    approval_history_path: str | Path = "reports/model/approval_history.json",
    approval_report_path: str | Path = "reports/model/approval_decision.md",
) -> dict[str, Any]:
    """Record APPROVE/REJECT and promote only after explicit APPROVE and stale checks."""
    normalized = decision.strip().upper()
    if normalized not in {"APPROVE", "REJECT"}:
        raise ValueError("Decision must be exactly APPROVE or REJECT")
    report = load_pending_report(report_path)
    client = MlflowClient(tracking_uri=tracking_uri)
    candidate = report["candidate"]
    champion_before = _current_champion(client, model_name)
    if str(champion_before.version) != str(report["champion"]["model_version"]):
        raise ValueError("Stale approval: current Champion changed since the report was created")
    candidate_version = client.get_model_version(model_name, str(candidate["model_version"]))
    if candidate_version.run_id != candidate["run_id"]:
        raise ValueError("Candidate run/version identity no longer matches the report")

    timestamp = datetime.now(timezone.utc).isoformat()
    result = "REJECTED"
    if normalized == "APPROVE":
        if report["comparison"].get("decision") != "PROMOTE":
            raise ValueError("Candidate failed automated promotion criteria")
        client.set_registered_model_alias(model_name, "champion", str(candidate_version.version))
        result = PROMOTED
        state = APPROVED
    else:
        state = REJECTED

    record = {
        "candidate_run_id": candidate["run_id"],
        "candidate_version": str(candidate_version.version),
        "champion_version_before": str(champion_before.version),
        "decision": state,
        "operator": operator,
        "timestamp": timestamp,
        "reason": reason or ("Approved by operator" if state == APPROVED else "Rejected by operator"),
        "promotion_result": result,
    }
    history_path = Path(approval_history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(record)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    report_path_obj = Path(approval_report_path)
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)
    report_path_obj.write_text(
        "\n".join(
            [
                "# Model Approval Decision",
                "",
                f"Recommended Decision: **{report['recommended_decision']}**",
                f"Human Decision: **{state}**",
                f"Final Result: **{result}**",
                "",
                f"Candidate version: `{candidate_version.version}`",
                f"Candidate run ID: `{candidate['run_id']}`",
                f"Champion before: `{champion_before.version}`",
                f"Operator: `{operator}`",
                f"Timestamp: `{timestamp}`",
                f"Reason: {record['reason']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report": report, "approval": record, "champion_after": str(_current_champion(client, model_name).version)}
