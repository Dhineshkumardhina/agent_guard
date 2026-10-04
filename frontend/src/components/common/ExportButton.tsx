import React from "react";
import { Download } from "lucide-react";

interface ExportButtonProps {
  data: unknown;
  filename: string;
  label?: string;
  format?: "json" | "csv";
  className?: string;
}

export const ExportButton: React.FC<ExportButtonProps> = ({
  data,
  filename,
  label = "Export",
  format = "json",
  className = "",
}) => {
  const handleExport = () => {
    let content = "";
    let mimeType = "application/json";
    let ext = "json";

    if (format === "json") {
      content = JSON.stringify(data, null, 2);
    } else if (format === "csv" && Array.isArray(data) && data.length > 0) {
      mimeType = "text/csv;charset=utf-8;";
      ext = "csv";
      const headers = Object.keys(data[0]);
      const rows = data.map((row) =>
        headers
          .map((field) => {
            const val = row[field];
            const escaped = String(val ?? "").replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(",")
      );
      content = [headers.join(","), ...rows].join("\n");
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `${filename}.${ext}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={handleExport}
      className={`inline-flex items-center gap-1.5 rounded border border-[#242b3d] bg-[#0c0e14] px-2.5 py-1.5 text-xs font-medium text-[#9aa4bf] transition-colors hover:bg-[#1b2030] hover:text-[#f0f3fa] ${className}`}
      title={`Export data as ${format.toUpperCase()}`}
    >
      <Download className="h-3.5 w-3.5 text-[#5e6984]" />
      <span>{label}</span>
    </button>
  );
};

export default ExportButton;
