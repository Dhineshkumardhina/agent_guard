import React from "react";

export interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "danger" | "info" | "outline";
  size?: "sm" | "md";
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "default",
  size = "sm",
  className = "",
}) => {
  const sizeClasses = size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs";

  const variantMap: Record<string, string> = {
    default: "bg-[#1b2030] text-[#9aa4bf] border border-[#242b3d]",
    success: "bg-[#064e3b]/30 text-[#34d399] border border-[#065f46]",
    warning: "bg-[#78350f]/30 text-[#fbbf24] border border-[#92400e]",
    danger: "bg-[#7f1d1d]/30 text-[#f87171] border border-[#991b1b]",
    info: "bg-[#0c4a6e]/30 text-[#38bdf8] border border-[#075985]",
    outline: "bg-transparent text-[#9aa4bf] border border-[#333d56]",
  };

  return (
    <span
      className={`inline-flex items-center font-mono font-medium rounded ${sizeClasses} ${variantMap[variant] || variantMap.default} ${className}`}
    >
      {children}
    </span>
  );
};

export const StatusBadge: React.FC<{ status: string; className?: string }> = ({
  status,
  className = "",
}) => {
  const s = status.toLowerCase();
  if (s === "completed" || s === "success" || s === "active") {
    return (
      <Badge variant="success" className={className}>
        {status}
      </Badge>
    );
  }
  if (s === "failed" || s === "failure") {
    return (
      <Badge variant="danger" className={className}>
        {status}
      </Badge>
    );
  }
  if (s === "running" || s === "degraded" || s === "watch") {
    return (
      <Badge variant="warning" className={className}>
        {status}
      </Badge>
    );
  }
  return (
    <Badge variant="default" className={className}>
      {status}
    </Badge>
  );
};

export const RiskBadge: React.FC<{ risk: string; className?: string }> = ({
  risk,
  className = "",
}) => {
  const r = risk.toUpperCase();
  if (r === "PREDICTED_CASCADE" || r === "CRITICAL") {
    return (
      <Badge variant="danger" className={className}>
        CASCADE
      </Badge>
    );
  }
  if (r === "HIGH_RISK") {
    return (
      <Badge variant="danger" className={className}>
        HIGH RISK
      </Badge>
    );
  }
  if (r === "WATCH") {
    return (
      <Badge variant="warning" className={className}>
        WATCH
      </Badge>
    );
  }
  return (
    <Badge variant="success" className={className}>
      NORMAL
    </Badge>
  );
};

export const ModelBadge: React.FC<{ model: string; className?: string }> = ({
  model,
  className = "",
}) => {
  const m = model.toLowerCase();
  let variant: "default" | "info" | "warning" | "success" = "default";
  if (m.includes("temporal") || m.includes("tgn")) {
    variant = "info";
  } else if (m.includes("gnn") || m.includes("gat") || m.includes("gcn")) {
    variant = "warning";
  } else if (m.includes("rule")) {
    variant = "default";
  } else {
    variant = "success";
  }
  return (
    <Badge variant={variant} className={className}>
      {model}
    </Badge>
  );
};

export const ExperimentBadge: React.FC<{ experimentId: string; className?: string }> = ({
  experimentId,
  className = "",
}) => {
  return (
    <span className={`font-mono text-xs text-[#9aa4bf] hover:text-[#f0f3fa] ${className}`}>
      {experimentId}
    </span>
  );
};
