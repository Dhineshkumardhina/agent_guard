import React, { useState } from "react";
import type { EventResponse, RunFailureResponse, PredictionResponse } from "../../types";
import { AlertTriangle, Bell, Zap, MessageSquare, Clock } from "lucide-react";

interface FailureTimelineProps {
  events: EventResponse[];
  failures: RunFailureResponse[];
  predictions?: PredictionResponse[];
  cascadeStep?: number;
  totalSteps?: number;
  durationSeconds?: number;
  onStepSelect?: (step: number) => void;
  className?: string;
}

export const FailureTimeline: React.FC<FailureTimelineProps> = ({
  events,
  failures,
  predictions = [],
  cascadeStep,
  totalSteps,
  onStepSelect,
  className = "",
}) => {
  const [selectedEvent, setSelectedEvent] = useState<EventResponse | null>(null);

  // Group events by discrete step index
  const maxStep = Math.max(
    1,
    totalSteps || 1,
    ...events.map((e) => e.step_idx),
    ...failures.map((f) => f.step_idx),
    cascadeStep || 0
  );

  // Identify first warning prediction
  const warningPred = predictions.find((p) => p.predicted_label === 1 || p.predicted_probability >= 0.5);
  const warningStep = warningPred?.step_idx ?? (warningPred ? 0 : undefined);
  const failureStep = cascadeStep ?? failures[0]?.step_idx;

  const leadTimeSteps =
    warningStep !== undefined && failureStep !== undefined && failureStep >= warningStep
      ? failureStep - warningStep
      : null;

  // Build step buckets
  const stepsArray = Array.from({ length: maxStep + 1 }, (_, i) => i);

  return (
    <div className={`rounded-lg border border-[#242b3d] bg-[#141721] p-4 ${className}`}>
      {/* Header Info Banner: Lead Time & Milestones */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-[#242b3d] pb-3 text-xs">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-[#9aa4bf]">
            <Clock className="h-4 w-4 text-[#5e6984]" />
            <span>Horizon Context: {maxStep} Steps</span>
          </div>

          {warningStep !== undefined && (
            <div className="flex items-center gap-1 font-mono text-[#f59e0b]">
              <Bell className="h-3.5 w-3.5" />
              <span>Warning Raised: Step {warningStep}</span>
            </div>
          )}

          {cascadeStep !== undefined && (
            <div className="flex items-center gap-1 font-mono text-[#ef4444]">
              <Zap className="h-3.5 w-3.5" />
              <span>Cascade Triggered: Step {cascadeStep}</span>
            </div>
          )}
        </div>

        {leadTimeSteps !== null && (
          <div className="rounded bg-[#1b253b] px-2.5 py-1 text-xs font-semibold text-[#60a5fa] border border-[#2563eb]/30">
            Advance Lead Time: +{leadTimeSteps} steps
          </div>
        )}
      </div>

      {/* Discrete Timeline Horizontal Scroller */}
      <div className="overflow-x-auto pb-4 pt-2">
        <div className="relative flex min-w-max items-center gap-6 px-4">
          {/* Background Connecting Axis Line */}
          <div className="absolute left-6 right-6 top-5 h-0.5 bg-[#242b3d]" />

          {stepsArray.map((step) => {
            const stepEvents = events.filter((e) => e.step_idx === step);
            const stepFailures = failures.filter((f) => f.step_idx === step);
            const stepPred = predictions.find((p) => p.step_idx === step);
            const isCascade = cascadeStep === step;
            const isWarning = warningStep === step;

            const hasFailure = stepFailures.length > 0 || isCascade;
            const hasEvents = stepEvents.length > 0;

            let nodeColor = "bg-[#1b2030] border-[#333d56]";
            if (isCascade) {
              nodeColor = "bg-[#7f1d1d] border-[#ef4444] text-[#fca5a5]";
            } else if (hasFailure) {
              nodeColor = "bg-[#450a0a] border-[#b91c1c] text-[#f87171]";
            } else if (isWarning) {
              nodeColor = "bg-[#451a03] border-[#f59e0b] text-[#fde68a]";
            } else if (hasEvents) {
              nodeColor = "bg-[#172554] border-[#3b82f6] text-[#93c5fd]";
            }

            return (
              <div
                key={step}
                className="relative z-10 flex flex-col items-center cursor-pointer group"
                onClick={() => {
                  if (onStepSelect) onStepSelect(step);
                  if (stepEvents.length > 0) setSelectedEvent(stepEvents[0]);
                }}
              >
                {/* Node Step Circle */}
                <div
                  className={`flex h-10 w-10 items-center justify-center rounded-full border-2 text-xs font-mono font-bold shadow-md transition-transform group-hover:scale-110 ${nodeColor}`}
                >
                  {isCascade ? (
                    <Zap className="h-4 w-4" />
                  ) : hasFailure ? (
                    <AlertTriangle className="h-4 w-4" />
                  ) : isWarning ? (
                    <Bell className="h-4 w-4" />
                  ) : hasEvents ? (
                    <MessageSquare className="h-3.5 w-3.5" />
                  ) : (
                    step
                  )}
                </div>

                {/* Step Index Label */}
                <span className="mt-1 text-[11px] font-mono text-[#5e6984] group-hover:text-[#f0f3fa]">
                  t={step}
                </span>

                {/* Badges / Micro Indicators below step */}
                <div className="mt-1 flex flex-col items-center gap-0.5">
                  {isCascade && (
                    <span className="rounded bg-[#ef4444]/20 px-1 py-0.2 text-[9px] font-bold text-[#ef4444] uppercase">
                      Cascade
                    </span>
                  )}
                  {!isCascade && hasFailure && (
                    <span className="rounded bg-[#b91c1c]/20 px-1 py-0.2 text-[9px] font-medium text-[#f87171]">
                      Fault
                    </span>
                  )}
                  {isWarning && (
                    <span className="rounded bg-[#f59e0b]/20 px-1 py-0.2 text-[9px] font-medium text-[#f59e0b]">
                      Warning
                    </span>
                  )}
                  {stepPred && (
                    <span className="font-mono text-[9px] text-[#9aa4bf]">
                      p={stepPred.predicted_probability.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Event Detail Drawer */}
      {selectedEvent && (
        <div className="mt-4 rounded-lg border border-[#242b3d] bg-[#0c0e14] p-3 text-xs">
          <div className="flex items-center justify-between border-b border-[#242b3d] pb-2">
            <span className="font-semibold text-[#f0f3fa]">
              Step {selectedEvent.step_idx} Event Inspection
            </span>
            <button
              onClick={() => setSelectedEvent(null)}
              className="text-[#9aa4bf] hover:text-[#f0f3fa]"
            >
              Close
            </button>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-[#9aa4bf]">
            <div>
              <span className="text-[#5e6984]">Routing:</span>{" "}
              <span className="text-[#f0f3fa] font-mono">
                {selectedEvent.source_agent || selectedEvent.sender} &rarr; {selectedEvent.target_agent || selectedEvent.receiver}
              </span>
            </div>
            <div>
              <span className="text-[#5e6984]">Type:</span>{" "}
              <span className="text-[#f0f3fa]">{selectedEvent.event_type}</span>
            </div>
            <div>
              <span className="text-[#5e6984]">Confidence:</span>{" "}
              <span className="text-[#f0f3fa] font-mono">
                {(selectedEvent.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-[#5e6984]">Quality:</span>{" "}
              <span className="text-[#f0f3fa] font-mono">
                {selectedEvent.output_quality.toFixed(3)}
              </span>
            </div>
            {selectedEvent.injected_fault && (
              <div className="col-span-2 text-[#fbbf24]">
                Injected Fault: <span className="font-mono">{selectedEvent.injected_fault}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FailureTimeline;
