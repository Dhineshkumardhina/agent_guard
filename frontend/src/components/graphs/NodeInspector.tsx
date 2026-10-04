import React from "react";
import { X, Cpu, AlertTriangle } from "lucide-react";
import { StatusBadge } from "../common/Badges";

interface NodeData {
  id: string;
  role?: string;
  is_active?: boolean;
  event_count?: number;
  message_count?: number;
  error_count?: number;
  retry_count?: number;
  average_latency?: number;
  average_confidence?: number;
  average_output_quality?: number;
  contradiction_rate?: number;
  recent_failure_count?: number;
  status?: string;
  [key: string]: unknown;
}

interface NodeInspectorProps {
  node: NodeData | null;
  onClose: () => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({ node, onClose }) => {
  if (!node) return null;

  const status = (node.status as string) || (node.recent_failure_count && node.recent_failure_count > 0 ? "failed" : "active");

  return (
    <div className="rounded-lg border border-[#333d56] bg-[#141721] p-4 text-xs shadow-lg">
      <div className="flex items-center justify-between border-b border-[#242b3d] pb-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-[#3b82f6]" />
          <span className="font-mono text-sm font-semibold text-[#f0f3fa]">
            {node.id}
          </span>
          <StatusBadge status={status} />
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
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Role</span>
          <div className="mt-0.5 font-medium text-[#f0f3fa] capitalize">
            {node.role || "Specialist Agent"}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Events Observed</span>
          <div className="mt-0.5 font-mono text-[#f0f3fa]">
            {node.event_count ?? node.message_count ?? 0}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Average Confidence</span>
          <div className="mt-0.5 font-mono text-[#f0f3fa]">
            {node.average_confidence !== undefined
              ? (Number(node.average_confidence) * 100).toFixed(1) + "%"
              : "N/A"}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Output Quality</span>
          <div className="mt-0.5 font-mono text-[#f0f3fa]">
            {node.average_output_quality !== undefined
              ? Number(node.average_output_quality).toFixed(3)
              : "N/A"}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Failures / Errors</span>
          <div className="mt-0.5 font-mono flex items-center gap-1.5 text-[#f87171]">
            <AlertTriangle className="h-3 w-3" />
            <span>{node.error_count ?? node.recent_failure_count ?? 0}</span>
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Retries / Timeouts</span>
          <div className="mt-0.5 font-mono text-[#fbbf24]">
            {node.retry_count ?? 0}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Mean Latency</span>
          <div className="mt-0.5 font-mono text-[#9aa4bf]">
            {node.average_latency !== undefined
              ? `${(Number(node.average_latency) * 1000).toFixed(0)} ms`
              : "-"}
          </div>
        </div>

        <div>
          <span className="text-[11px] text-[#5e6984] uppercase tracking-wider">Contradiction Rate</span>
          <div className="mt-0.5 font-mono text-[#9aa4bf]">
            {node.contradiction_rate !== undefined
              ? (Number(node.contradiction_rate) * 100).toFixed(1) + "%"
              : "0.0%"}
          </div>
        </div>
      </div>
    </div>
  );
};
