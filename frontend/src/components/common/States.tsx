import React from "react";
import { AlertCircle, Inbox, RefreshCw } from "lucide-react";

export const LoadingState: React.FC<{ message?: string; className?: string }> = ({
  message = "Loading research data...",
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-12 text-center text-[#9aa4bf] ${className}`}
    >
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#333d56] border-t-[#3b82f6]" />
      <span className="mt-3 text-xs font-medium tracking-wide uppercase">
        {message}
      </span>
    </div>
  );
};

export const EmptyState: React.FC<{
  title?: string;
  message?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}> = ({
  title = "No research records found",
  message = "No matching experiments, runs, or evaluations match current criteria.",
  icon,
  action,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-lg border border-dashed border-[#242b3d] p-12 text-center ${className}`}
    >
      <div className="text-[#5e6984] mb-3">
        {icon || <Inbox className="h-8 w-8" />}
      </div>
      <h4 className="text-sm font-semibold text-[#f0f3fa]">{title}</h4>
      <p className="mt-1 max-w-sm text-xs text-[#9aa4bf]">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
};

export const ErrorState: React.FC<{
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}> = ({
  title = "Failed to load research data",
  message,
  onRetry,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-lg border border-[#7f1d1d]/40 bg-[#7f1d1d]/10 p-8 text-center ${className}`}
    >
      <AlertCircle className="h-7 w-7 text-[#f87171] mb-2" />
      <h4 className="text-sm font-semibold text-[#f87171]">{title}</h4>
      <p className="mt-1 max-w-md text-xs text-[#fca5a5] font-mono">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 inline-flex items-center gap-1.5 rounded border border-[#991b1b] bg-[#7f1d1d]/40 px-3 py-1.5 text-xs font-medium text-[#fee2e2] transition-colors hover:bg-[#7f1d1d]/60"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Retry Request
        </button>
      )}
    </div>
  );
};
