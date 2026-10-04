import React from "react";
import { X, ArrowRight, Share2, AlertCircle } from "lucide-react";

interface EdgeData {
  source: string;
  target: string;
  key?: string;
  interaction_count?: number;
  message_count?: number;
  average_latency?: number;
  average_confidence?: number;
  average_message_length?: number;
  contradiction_rate?: number;
  error_count?: number;
  timeout_count?: number;
  [key: string]: unknown;
}

interface EdgeInspectorProps {
  edge: EdgeData | null;
  onClose: () => void;
}

export const EdgeInspector: React.FC<EdgeInspectorProps> = ({ edge, onClose }) => {
  if (!edge) return null;

  const count = edge.interaction_count ?? edge.message_count ?? 1;
  const contradiction = Number(edge.contradiction_rate ?? 0);
  const errors = edge.error_count ?? 0;

  return (
    <div className="rounded-lg border border-[#333d56] bg-[#141721] p-4 text-xs shadow-lg">
      <div className="flex items-center justify-between border-b border-[#242b3d] pb-2.5">
        <div className="flex items-center gap-2 font-mono text-sm font-semibold text-[#f0f3fa]">
          <Share2 className="h-4 w-4 text-[#6366f1]" />
          <span>{edge.source}</span>
          <ArrowRight className="h-3.5 w-3.5 text-[#5e6984]" />
          <span>{edge.target}</span>
        </div>
        <button
          onClick={onClose}
          className="rounded p-1 text-[#9aa4bf] hover:bg-[#1b2030] hover:text-[#f0f3fa]"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Interaction Count</span>
          <div className="mt-0.5 font-mono text-[#f0f3fa]">{count}</div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Contradiction Rate</span>
          <div className="mt-0.5 font-mono">
            {contradiction > 0.2 ? (
              <span className="text-[#f87171] font-semibold">
                {(contradiction * 100).toFixed(1)}%
              </span>
            ) : (
              <span className="text-[#34d399]">
                {(contradiction * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Average Latency</span>
          <div className="mt-0.5 font-mono text-[#9aa4bf]">
            {edge.average_latency !== undefined
              ? `${(Number(edge.average_latency) * 1000).toFixed(0)} ms`
              : "-"}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Errors Encountered</span>
          <div className="mt-0.5 font-mono">
            {errors > 0 ? (
              <span className="text-[#f87171] flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                {errors}
              </span>
            ) : (
              <span className="text-[#9aa4bf]">0</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
