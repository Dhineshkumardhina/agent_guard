"""TelemetryCollector: Research-Grade Multi-Agent Event Stream Management.

Receives, validates, timestamps, correlates, persists, and exports multi-agent execution events.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import csv
import time
from uuid import uuid4

from sqlalchemy.orm import Session

from ml.telemetry.schemas import (
    AgentTelemetryEvent,
    TelemetryEventType,
    SUPPORTED_EVENT_TYPES,
)
from backend.app.database.models import Event as DBEvent


class TelemetryCollector:
    """Central collector for telemetry and event streams across multi-agent runs."""

    def __init__(
        self,
        active_run_id: Optional[str] = None,
        strict_event_types: bool = False,
    ) -> None:
        """Initialize telemetry collector.
        
        Args:
            active_run_id: Optional default run ID to associate incoming events with.
            strict_event_types: If True, rejects any event_type outside TelemetryEventType.
        """
        self.active_run_id: Optional[str] = active_run_id
        self.strict_event_types: bool = strict_event_types
        self._events: List[AgentTelemetryEvent] = []
        self._step_counter: int = 0

    @property
    def events(self) -> List[AgentTelemetryEvent]:
        """All collected events in chronological order."""
        return list(self._events)

    def validate_event(self, event: AgentTelemetryEvent) -> bool:
        """Validate an event against telemetry integrity criteria.
        
        Raises:
            ValueError: If validation fails.
        """
        if not event.source_agent or not isinstance(event.source_agent, str):
            raise ValueError(f"Event must have a valid string source_agent, got: {event.source_agent!r}")
        if not event.target_agent or not isinstance(event.target_agent, str):
            raise ValueError(f"Event must have a valid string target_agent, got: {event.target_agent!r}")

        # Check event_type
        if self.strict_event_types:
            clean_type = str(event.event_type).lower().strip()
            if clean_type not in SUPPORTED_EVENT_TYPES:
                raise ValueError(
                    f"Invalid event_type '{event.event_type}'. Must be one of: {sorted(list(SUPPORTED_EVENT_TYPES))}"
                )

        # Check probability bounds
        if event.confidence is not None and not (0.0 <= event.confidence <= 1.0):
            raise ValueError(f"Confidence score {event.confidence} must be in [0.0, 1.0]")
        if event.contradiction_score is not None and not (0.0 <= event.contradiction_score <= 1.0):
            raise ValueError(f"Contradiction score {event.contradiction_score} must be in [0.0, 1.0]")
        if event.output_quality is not None and not (0.0 <= event.output_quality <= 1.0):
            raise ValueError(f"Output quality score {event.output_quality} must be in [0.0, 1.0]")

        return True

    def assign_id(self, event: AgentTelemetryEvent) -> None:
        """Assign unique event ID if missing."""
        if not event.event_id:
            event.event_id = str(uuid4())

    def timestamp_event(self, event: AgentTelemetryEvent, current_time: Optional[float] = None) -> None:
        """Ensure monotonic/valid simulation timestamp is present."""
        if event.timestamp is None:
            if current_time is not None:
                event.timestamp = current_time
            elif self._events:
                # Increment from previous event
                event.timestamp = round(self._events[-1].timestamp + 0.1, 4)
            else:
                event.timestamp = 0.0

    def associate_run(self, event: AgentTelemetryEvent, run_id: Optional[str] = None) -> None:
        """Bind event to designated or active run ID."""
        target_run_id = run_id or self.active_run_id
        if target_run_id:
            event.run_id = target_run_id
        elif not event.run_id:
            raise ValueError("Event has no run_id and collector has no active_run_id set.")

    def receive_event(
        self,
        event: Union[AgentTelemetryEvent, Dict[str, Any], Any],
        run_id: Optional[str] = None,
    ) -> AgentTelemetryEvent:
        """Ingest, validate, normalize, and buffer an incoming event.
        
        Args:
            event: An AgentTelemetryEvent, SimulationMessage, or dictionary.
            run_id: Optional explicit run identifier to associate.
            
        Returns:
            The validated AgentTelemetryEvent.
        """
        # Convert dictionary or SimulationMessage to AgentTelemetryEvent
        if isinstance(event, dict):
            telemetry_event = AgentTelemetryEvent(**event)
        elif hasattr(event, "to_telemetry_event"):
            telemetry_event = event.to_telemetry_event()
        elif isinstance(event, AgentTelemetryEvent):
            telemetry_event = event
        else:
            raise TypeError(f"Unsupported event object type: {type(event)}")

        # 1. Assign ID if missing
        self.assign_id(telemetry_event)

        # 2. Associate Run
        self.associate_run(telemetry_event, run_id=run_id)

        # 3. Assign Timestamp if missing
        self.timestamp_event(telemetry_event)

        # 4. Assign Step index if not set
        if telemetry_event.step_idx is None:
            telemetry_event.step_idx = self._step_counter
        self._step_counter += 1

        # 5. Validate integrity
        self.validate_event(telemetry_event)

        # 6. Buffer event maintain chronological order
        self._events.append(telemetry_event)
        # Keep events sorted chronologically by timestamp, then step_idx
        self._events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

        return telemetry_event

    def receive_batch(
        self,
        events: List[Any],
        run_id: Optional[str] = None,
    ) -> List[AgentTelemetryEvent]:
        """Ingest a sequence of events."""
        return [self.receive_event(e, run_id=run_id) for e in events]

    def get_events(
        self,
        run_id: Optional[str] = None,
        source_agent: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[AgentTelemetryEvent]:
        """Query buffered events matching optional filter criteria."""
        filtered = self._events
        if run_id:
            filtered = [e for e in filtered if e.run_id == run_id]
        if source_agent:
            filtered = [e for e in filtered if e.source_agent == source_agent]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        return filtered

    def clear(self) -> None:
        """Clear collector memory buffer."""
        self._events.clear()
        self._step_counter = 0

    def persist_to_db(self, session: Session, run_id: Optional[str] = None) -> int:
        """Persist collected events into the relational database.
        
        Returns:
            Count of newly persisted events.
        """
        target_events = self.get_events(run_id=run_id)
        persisted_count = 0

        for ev in target_events:
            existing = session.query(DBEvent).filter(DBEvent.id == ev.event_id).first()
            if not existing:
                db_event = DBEvent(
                    id=ev.event_id,
                    run_id=ev.run_id,
                    step_idx=ev.step_idx or 0,
                    timestamp=round(ev.timestamp or 0.0, 4),
                    source_agent=ev.source_agent,
                    target_agent=ev.target_agent,
                    event_type=ev.event_type,
                    message_length=ev.message_length,
                    token_count=ev.token_count,
                    latency=round(ev.latency or 0.0, 4),
                    confidence=round(ev.confidence or 1.0, 4),
                    output_quality=round(ev.output_quality or 1.0, 4),
                    contradiction_score=round(ev.contradiction_score or 0.0, 4),
                    tool_used=ev.tool_used,
                    tool_success=ev.tool_success,
                    tool_error=ev.tool_error,
                    retry_count=ev.retry_count,
                    injected_fault=ev.injected_fault,
                    error_type=ev.error_type,
                    failure_label=ev.failure_label,
                    downstream_failure=ev.downstream_failure,
                    metadata_json=ev.metadata,
                )
                session.add(db_event)
                persisted_count += 1

        session.commit()
        return persisted_count

    def export(
        self,
        filepath: Union[str, Path],
        format: str = "jsonl",
        run_id: Optional[str] = None,
    ) -> Path:
        """Export raw telemetry events to designated file format.
        
        Raw events remain strictly unchanged.
        
        Supported formats:
        - 'jsonl' (JSON Lines)
        - 'csv' (Comma Separated Values)
        - 'parquet' (requires pyarrow)
        
        Args:
            filepath: Destination file path.
            format: 'jsonl', 'csv', or 'parquet'.
            run_id: Optional filter for specific run.
            
        Returns:
            Path to exported file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        events_to_export = self.get_events(run_id=run_id)

        clean_format = format.lower().strip()

        if clean_format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for ev in events_to_export:
                    f.write(json.dumps(ev.to_dict()) + "\n")

        elif clean_format == "csv":
            if not events_to_export:
                # Write empty CSV with standard headers
                headers = list(AgentTelemetryEvent(source_agent="a", target_agent="b").to_dict().keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
            else:
                rows = []
                for ev in events_to_export:
                    d = ev.to_dict()
                    # Flatten metadata dict to JSON string for CSV compatibility
                    d["metadata"] = json.dumps(d.get("metadata", {}))
                    rows.append(d)
                headers = list(rows[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)

        elif clean_format == "parquet":
            try:
                import pandas as pd
                rows = []
                for ev in events_to_export:
                    d = ev.to_dict()
                    d["metadata"] = json.dumps(d.get("metadata", {}))
                    rows.append(d)
                df = pd.DataFrame(rows)
                df.to_parquet(path, index=False)
            except ImportError as e:
                raise ImportError(
                    "Parquet export requires 'pyarrow' or 'fastparquet' to be installed. "
                    "Install with: pip install pyarrow"
                ) from e
        else:
            raise ValueError(f"Unsupported export format: '{format}'. Supported formats: ['jsonl', 'csv', 'parquet']")

        return path
