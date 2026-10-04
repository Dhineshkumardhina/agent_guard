import React from "react";

interface ChartContainerProps {
  title?: string;
  subtitle?: string;
  height?: number;
  action?: React.ReactNode;
  children: React.ReactElement;
  className?: string;
}

export const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  subtitle,
  height = 300,
  action,
  children,
  className = "",
}) => {
  return (
    <div
      className={`rounded-lg border border-[#242b3d] bg-[#141721] p-4 shadow-sm ${className}`}
    >
      {(title || subtitle || action) && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-[#242b3d] pb-3">
          <div>
            {title && (
              <h4 className="text-sm font-semibold tracking-wide text-[#f0f3fa]">
                {title}
              </h4>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-[#9aa4bf]">{subtitle}</p>
            )}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}

      <div style={{ width: "100%", height }}>
        {children}
      </div>
    </div>
  );
};

export default ChartContainer;
