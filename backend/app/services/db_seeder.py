"""Database seeder to sync real research runs, agents, failures, and predictions into SQLite."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.database.models import (
    Run,
    Agent,
    Event,
    Failure,
    Prediction,
    Experiment,
    ModelResult,
)

logger = get_logger(__name__)


def seed_research_database(db: Session, force: bool = False) -> None:
    """Populate database with verified runs, agents, failures, predictions, and model results."""
    # Check if runs are already populated beyond demo runs
    existing_runs_count = db.query(Run).count()
    if existing_runs_count >= 10 and not force:
        logger.info("Database already seeded with %d runs, skipping seed", existing_runs_count)
        return

    logger.info("Seeding research database from processed datasets and results...")

    # 1. Load samples from all_samples.parquet
    parquet_path = settings.BASE_DIR / "data" / "processed" / "agentguard_generalization_v1" / "all_samples.parquet"
    if not parquet_path.exists():
        logger.warning("Dataset parquet file not found at %s", parquet_path)
        return

    table = pq.read_table(str(parquet_path))
    data = table.to_pydict()

    # Aggregate by run_id
    run_records = {}
    total_rows = len(data["run_id"])
    for i in range(total_rows):
        rid = data["run_id"][i]
        label = data["label"][i]
        step = data["step_idx"][i]
        ts = data["timestamp"][i]

        if rid not in run_records:
            run_records[rid] = {
                "id": rid,
                "task_type": data["task_type"][i],
                "topology": data["topology"][i],
                "num_agents": data["number_of_agents"][i],
                "random_seed": data["random_seed"][i],
                "dataset_id": data["dataset_version"][i],
                "has_cascading_failure": False,
                "cascading_failure_step": None,
                "duration_seconds": 0.0,
                "failure_type": data["failure_type"][i],
                "failure_level": data["failure_level"][i],
            }

        if label > 0:
            run_records[rid]["has_cascading_failure"] = True
            current_step = run_records[rid]["cascading_failure_step"]
            if current_step is None or step < current_step:
                run_records[rid]["cascading_failure_step"] = step

        if ts > run_records[rid]["duration_seconds"]:
            run_records[rid]["duration_seconds"] = float(ts)

    # Insert or update Runs
    agent_roles = [
        ("planner", "Task planning and decomposition"),
        ("researcher", "Information gathering and analysis"),
        ("analyst", "Synthesizing and quantitative assessment"),
        ("verifier", "Verification, schema validation, and constraint checking"),
        ("decision", "Final consensus determination and routing"),
        ("executor", "Tool execution and environment interfacing"),
        ("monitor", "Runtime observability and safety telemetry"),
        ("coordinator", "Inter-agent synchronization and routing"),
        ("critic", "Refinement and feedback generation"),
        ("librarian", "Persistent knowledge and context retrieval"),
        ("worker", "General specialized task execution"),
        ("supervisor", "Execution management and error handling"),
    ]

    base_time = datetime(2026, 10, 4, 2, 15, tzinfo=timezone.utc)

    for rid, rdata in run_records.items():
        existing_run = db.query(Run).filter(Run.id == rid).first()
        if not existing_run:
            run = Run(
                id=rid,
                task_type=rdata["task_type"],
                topology=rdata["topology"],
                num_agents=rdata["num_agents"],
                duration_seconds=max(rdata["duration_seconds"], 1.5),
                has_cascading_failure=rdata["has_cascading_failure"],
                cascading_failure_step=rdata["cascading_failure_step"],
                random_seed=rdata["random_seed"],
                dataset_id=rdata["dataset_id"],
                created_at=base_time,
            )
            db.add(run)

            # Add agents for this run
            num_agents = rdata["num_agents"]
            for a_idx in range(num_agents):
                role_idx = a_idx % len(agent_roles)
                role_name, role_desc = agent_roles[role_idx]
                agent_name = f"{role_name}_{a_idx + 1}"
                agent_id = f"{rid}_{agent_name}"
                agent = Agent(
                    id=agent_id,
                    run_id=rid,
                    agent_name=agent_name,
                    role=role_name,
                    status="failed" if (rdata["has_cascading_failure"] and a_idx == 0) else "active",
                    created_at=base_time,
                )
                db.add(agent)

            # Add failure record if cascading failure occurred
            if rdata["has_cascading_failure"]:
                fail_id = f"fail_{rid}"
                originating = f"{agent_roles[0][0]}_1"
                affected = [f"{agent_roles[i % len(agent_roles)][0]}_{i + 1}" for i in range(min(num_agents, 4))]
                failure = Failure(
                    id=fail_id,
                    run_id=rid,
                    step_idx=rdata["cascading_failure_step"] or 1,
                    failure_level=int(rdata.get("failure_level", 3) or 3),
                    originating_agent=originating,
                    affected_agents=affected,
                    failure_type=rdata.get("failure_type") or "cascading_error",
                    description=f"Cascading propagation initiated by {originating} impacting {len(affected)} agents.",
                    created_at=base_time,
                )
                db.add(failure)

    # Also seed raw demo run events if present
    raw_telemetry = settings.BASE_DIR / "data" / "raw" / "run_telemetry_demo_001.jsonl"
    if raw_telemetry.exists():
        raw_run_id = "run_telemetry_demo_001"
        if not db.query(Run).filter(Run.id == raw_run_id).first():
            demo_run = Run(
                id=raw_run_id,
                task_type="research",
                topology="pipeline",
                num_agents=5,
                duration_seconds=1.0,
                has_cascading_failure=False,
                random_seed=42,
                dataset_id="raw_demo",
                created_at=base_time,
            )
            db.add(demo_run)
            for a_idx in range(5):
                rname = agent_roles[a_idx % len(agent_roles)][0]
                agent = Agent(
                    id=f"{raw_run_id}_{rname}_{a_idx + 1}",
                    run_id=raw_run_id,
                    agent_name=f"{rname}_{a_idx + 1}",
                    role=rname,
                    status="active",
                    created_at=base_time,
                )
                db.add(agent)

        with open(raw_telemetry, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev_data = json.loads(line)
                eid = ev_data.get("event_id")
                if eid and not db.query(Event).filter(Event.id == eid).first():
                    event = Event(
                        id=eid,
                        run_id=ev_data.get("run_id", raw_run_id),
                        step_idx=int(ev_data.get("step_idx", 0)),
                        timestamp=float(ev_data.get("timestamp", 0.0)),
                        source_agent=ev_data.get("source_agent", "unknown"),
                        target_agent=ev_data.get("target_agent", "unknown"),
                        event_type=ev_data.get("event_type", "message"),
                        message_length=int(ev_data.get("message_length", 0)),
                        token_count=int(ev_data.get("token_count", 0)),
                        latency=float(ev_data.get("latency", 0.0)),
                        confidence=float(ev_data.get("confidence", 1.0)),
                        output_quality=float(ev_data.get("output_quality", 1.0)),
                        contradiction_score=float(ev_data.get("contradiction_score", 0.0)),
                        tool_used=ev_data.get("tool_used"),
                        tool_success=ev_data.get("tool_success"),
                        tool_error=bool(ev_data.get("error_type") is not None),
                        retry_count=int(ev_data.get("retry_count", 0)),
                        injected_fault=ev_data.get("injected_fault"),
                        error_type=ev_data.get("error_type"),
                        failure_label=int(ev_data.get("failure_label", 0)),
                        downstream_failure=bool(ev_data.get("downstream_failure", False)),
                        metadata_json=ev_data.get("metadata", {}),
                        created_at=base_time,
                    )
                    db.add(event)

    db.commit()
    logger.info("Inserted %d runs and associated agents/failures into DB", len(run_records))

    # 2. Seed Predictions from results/baselines/
    pred_count = db.query(Prediction).count()
    if pred_count == 0:
        _seed_predictions(db)

    # 3. Seed Experiments & Model Results from unified_evaluation_metrics.json
    exp_count = db.query(Experiment).count()
    if exp_count == 0:
        _seed_evaluations_and_experiments(db)


def _seed_predictions(db: Session) -> None:
    """Seed prediction inferences from saved prediction json files."""
    results_dir = settings.BASE_DIR / "results" / "baselines"
    pred_files = list(results_dir.glob("**/predictions.json"))

    created_predictions = 0
    base_time = datetime(2026, 10, 4, 3, 0, tzinfo=timezone.utc)

    seen_pids = set()
    for pfile in pred_files:
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else list(data.values())
            for item in items:
                model_name = item.get("model_name", "temporal_gnn")
                horizon = int(item.get("horizon", 1))
                raw_sample_id = item.get("sample_id") or f"s_{created_predictions}"
                pid = f"{model_name}_{raw_sample_id}_k{horizon}"
                if pid in seen_pids:
                    pid = f"{pid}_{created_predictions}"
                seen_pids.add(pid)
                existing = db.query(Prediction).filter(Prediction.id == pid).first()
                if not existing:
                    prob = float(item.get("predicted_probability", 0.0))
                    risk = "PREDICTED_CASCADE" if prob >= 0.75 else ("HIGH_RISK" if prob >= 0.5 else ("WATCH" if prob >= 0.25 else "NORMAL"))
                    pred = Prediction(
                        id=pid,
                        run_id=item.get("run_id", "unknown_run"),
                        step_idx=int(item.get("step_idx", 0) or 0),
                        model_name=model_name,
                        horizon_k=int(item.get("horizon", 1)),
                        predicted_probability=prob,
                        risk_level=risk,
                        ground_truth=item.get("true_label") or item.get("ground_truth"),
                        lead_time=item.get("lead_time"),
                        explanation_json={},
                        created_at=base_time,
                    )
                    db.add(pred)
                    created_predictions += 1
        except Exception as e:
            logger.warning("Error reading prediction file %s: %s", pfile, e)

    db.commit()
    logger.info("Seeded %d predictions into DB", created_predictions)


def _seed_evaluations_and_experiments(db: Session) -> None:
    """Seed experiment and model evaluation records."""
    metrics_path = settings.BASE_DIR / "results" / "evaluation" / "metrics" / "unified_evaluation_metrics.json"
    if not metrics_path.exists():
        return

    base_time = datetime(2026, 10, 4, 4, 0, tzinfo=timezone.utc)

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # Unique experiments
        exp_map = {}
        for entry in eval_data:
            mname = entry.get("model_name")
            if mname not in exp_map:
                exp_id = f"exp_eval_{mname.lower().replace(' ', '_').replace('-', '_')}"
                exp = Experiment(
                    id=exp_id,
                    name=f"{mname} Benchmark Evaluation",
                    description=f"Evaluation benchmark for {mname} on AgentGuard dataset",
                    config={"model": mname, "dataset_version": entry.get("dataset_version")},
                    status="completed",
                    created_at=base_time,
                    updated_at=base_time,
                )
                db.add(exp)
                exp_map[mname] = exp_id

            # Add ModelResult
            m_res_id = f"res_{entry['experiment_id']}"
            m_res = ModelResult(
                id=m_res_id,
                experiment_id=exp_map[mname],
                model_name=mname,
                horizon_k=entry.get("prediction_horizon", 1),
                precision=float(entry.get("precision", 0.0)),
                recall=float(entry.get("recall", 0.0)),
                f1=float(entry.get("f1", 0.0)),
                auroc=float(entry.get("auroc", 0.0)),
                auprc=float(entry.get("auprc", 0.0)),
                false_alarm_rate=float(entry.get("false_alarm_rate", 0.0)),
                mean_lead_time=float(entry.get("mean_lead_time", 0.0)),
                median_lead_time=float(entry.get("median_lead_time", 0.0)),
                metrics_json=entry,
                created_at=base_time,
            )
            db.add(m_res)

        db.commit()
        logger.info("Seeded %d experiments and model results into DB", len(eval_data))
    except Exception as e:
        logger.warning("Error seeding evaluations: %s", e)
