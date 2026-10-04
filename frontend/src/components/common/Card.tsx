import React from "react";

interface CardProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  action,
  icon,
  children,
  className = "",
  bodyClassName = "",
}) => {
  return (
    <div
      className={`rounded-lg border border-[#242b3d] bg-[#141721] shadow-sm ${className}`}
    >
      {(title || action || icon) && (
        <div className="flex items-center justify-between border-b border-[#242b3d] px-4 py-3">
          <div className="flex items-center gap-2.5">
            {icon && <div className="shrink-0">{icon}</div>}
            <div>
              {typeof title === "string" ? (
                <h3 className="text-sm font-semibold tracking-wide text-[#f0f3fa]">
                  {title}
                </h3>
              ) : (
                title
              )}
              {subtitle && (
                <p className="mt-0.5 text-xs text-[#9aa4bf]">{subtitle}</p>
              )}
            </div>
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      <div className={`p-4 ${bodyClassName}`}>{children}</div>
    </div>
  );
};

export default Card;
