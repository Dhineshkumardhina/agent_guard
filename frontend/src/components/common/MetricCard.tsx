import React from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  trend?: "neutral" | "positive" | "negative";
  status?: "success" | "warning" | "danger" | "info" | "neutral";
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  badge,
  icon,
  status,
  className = "",
}) => {
  const statusColor =
    status === "success"
      ? "text-emerald-400"
      : status === "warning"
      ? "text-amber-400"
      : status === "danger"
      ? "text-rose-400"
      : status === "info"
      ? "text-cyan-400"
      : "text-[#f0f3fa]";

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-[#242b3d] bg-[#141721] p-4 transition-colors hover:border-[#333d56] ${className}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-[#9aa4bf]">
          {label}
        </span>
        {icon && <span className="text-[#5e6984]">{icon}</span>}
      </div>

      <div className="mt-2 flex items-baseline gap-2">
        <span className={`font-mono text-2xl font-semibold tracking-tight ${statusColor}`}>
          {value}
        </span>
        {badge && <span>{badge}</span>}
      </div>

      {subtext && (
        <div className="mt-1.5 text-xs text-[#5e6984]">{subtext}</div>
      )}
    </div>
  );
};

export default MetricCard;
