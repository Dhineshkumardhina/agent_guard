import React from "react";
import { Filter, X } from "lucide-react";

export interface FilterOption {
  label: string;
  value: string;
}

export interface FilterField {
  id?: string;
  key?: string;
  label: string;
  type?: "select" | "text";
  options?: FilterOption[];
  value: string;
  placeholder?: string;
}

interface FilterBarProps {
  fields: FilterField[];
  onFilterChange?: (key: string, value: string) => void;
  onChange?: (key: string, value: string) => void;
  onReset: () => void;
  className?: string;
  children?: React.ReactNode;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  fields,
  onFilterChange,
  onChange,
  onReset,
  className = "",
  children,
}) => {
  const triggerChange = (key: string, val: string) => {
    if (onChange) onChange(key, val);
    if (onFilterChange) onFilterChange(key, val);
  };

  const hasActiveFilters = fields.some(
    (f) => f.value !== "" && f.value !== "all" && f.value !== undefined
  );

  return (
    <div
      className={`flex flex-wrap items-center gap-3 rounded-lg border border-[#242b3d] bg-[#141721] p-3 text-xs ${className}`}
    >
      <div className="flex items-center gap-1.5 text-[#9aa4bf]">
        <Filter className="h-3.5 w-3.5" />
        <span className="font-semibold uppercase tracking-wider text-[11px]">
          Filters:
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2 flex-1">
        {fields.map((field) => {
          const fieldKey = field.id || field.key || field.label;
          return (
            <div key={fieldKey} className="flex items-center gap-1.5">
              <span className="text-[#5e6984]">{field.label}:</span>
              <select
                value={field.value}
                onChange={(e) => triggerChange(fieldKey, e.target.value)}
                className="rounded border border-[#242b3d] bg-[#0c0e14] px-2 py-1 text-xs text-[#f0f3fa] focus:border-[#3b82f6] focus:outline-none"
              >
                {field.placeholder && (
                  <option value="">{field.placeholder}</option>
                )}
                {(field.options || []).map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          );
        })}
      </div>

      {hasActiveFilters && (
        <button
          onClick={onReset}
          className="flex items-center gap-1 rounded border border-[#242b3d] bg-[#1a1e2b] px-2 py-1 text-xs text-[#9aa4bf] hover:bg-[#242b3d] hover:text-[#f0f3fa]"
        >
          <X className="h-3 w-3" />
          <span>Reset</span>
        </button>
      )}

      {children && <div className="ml-auto">{children}</div>}
    </div>
  );
};

export default FilterBar;
