import React, { useState, useMemo } from "react";
import { ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Search } from "lucide-react";
import { EmptyState } from "./States";

export interface Column<T> {
  key?: string;
  header: string;
  accessor?: (item: T) => React.ReactNode;
  render?: (item: T, index: number) => React.ReactNode;
  sortBy?: (item: T) => any;
  sortable?: boolean;
  align?: "left" | "center" | "right";
  width?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor?: (item: T) => string;
  keyField?: keyof T | string;
  searchableKey?: keyof T | ((item: T) => string);
  searchPlaceholder?: string;
  defaultSortKey?: string;
  defaultSortOrder?: "asc" | "desc";
  pageSize?: number;
  emptyMessage?: string;
  className?: string;
  onRowClick?: (item: T) => void;
  headerAction?: React.ReactNode;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data,
  keyExtractor,
  keyField,
  searchableKey,
  searchPlaceholder = "Search table...",
  defaultSortKey,
  defaultSortOrder = "asc",
  pageSize = 15,
  emptyMessage = "No matching records found.",
  className = "",
  onRowClick,
  headerAction,
}: DataTableProps<T>) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortIndex, setSortIndex] = useState<number | undefined>(() => {
    if (defaultSortKey) {
      const idx = columns.findIndex((c) => (c.key || c.header) === defaultSortKey);
      return idx >= 0 ? idx : undefined;
    }
    return undefined;
  });
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">(defaultSortOrder);
  const [currentPage, setCurrentPage] = useState(1);

  // Filter
  const filteredData = useMemo(() => {
    if (!searchTerm) return data;
    const term = searchTerm.toLowerCase();

    return data.filter((item) => {
      if (searchableKey) {
        let val: unknown;
        if (typeof searchableKey === "function") {
          val = searchableKey(item);
        } else {
          val = item[searchableKey];
        }
        return String(val ?? "").toLowerCase().includes(term);
      }

      // Default: check all string/number fields of the row
      return Object.values(item).some((v) =>
        String(v ?? "").toLowerCase().includes(term)
      );
    });
  }, [data, searchTerm, searchableKey]);

  // Sort
  const sortedData = useMemo(() => {
    if (sortIndex === undefined || !columns[sortIndex]) return filteredData;
    const col = columns[sortIndex];

    return [...filteredData].sort((a, b) => {
      let aVal: any;
      let bVal: any;

      if (col.sortBy) {
        aVal = col.sortBy(a);
        bVal = col.sortBy(b);
      } else if (col.key && a[col.key] !== undefined) {
        aVal = a[col.key];
        bVal = b[col.key];
      } else {
        return 0;
      }

      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortOrder === "asc" ? aVal - bVal : bVal - aVal;
      }
      return sortOrder === "asc"
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
  }, [filteredData, sortIndex, sortOrder, columns]);

  // Paginate
  const totalPages = Math.max(1, Math.ceil(sortedData.length / pageSize));
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const handleSort = (idx: number) => {
    if (sortIndex === idx) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortIndex(idx);
      setSortOrder("asc");
    }
  };

  const getItemKey = (item: T, index: number): string => {
    if (keyExtractor) return keyExtractor(item);
    if (keyField && item[keyField as string]) return String(item[keyField as string]);
    if (item.id) return String(item.id);
    if (item.run_id) return String(item.run_id);
    if (item.prediction_id) return String(item.prediction_id);
    if (item.experiment_id) return String(item.experiment_id);
    if (item.explanation_id) return String(item.explanation_id);
    return `row-${index}`;
  };

  return (
    <div className={`overflow-hidden rounded-lg border border-[#242b3d] bg-[#141721] ${className}`}>
      {(searchPlaceholder || headerAction) && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#242b3d] px-4 py-2.5">
          <div className="relative min-w-[240px] max-w-sm flex-1">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#5e6984]" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              placeholder={searchPlaceholder}
              className="w-full rounded border border-[#242b3d] bg-[#0c0e14] py-1.5 pl-8 pr-3 text-xs text-[#f0f3fa] placeholder-[#5e6984] focus:border-[#3b82f6] focus:outline-none"
            />
          </div>
          {headerAction && <div className="flex items-center gap-2">{headerAction}</div>}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-[#242b3d] bg-[#0f121a] text-[11px] font-semibold uppercase tracking-wider text-[#9aa4bf]">
            <tr>
              {columns.map((col, idx) => {
                const isSorted = sortIndex === idx;
                const canSort = col.sortable !== false && (col.sortBy || col.key);
                return (
                  <th
                    key={col.key || `col-${idx}`}
                    scope="col"
                    style={{ width: col.width }}
                    onClick={() => canSort && handleSort(idx)}
                    className={`px-4 py-2.5 ${canSort ? "cursor-pointer select-none hover:text-[#f0f3fa]" : ""} ${
                      col.align === "right" ? "text-right" : col.align === "center" ? "text-center" : "text-left"
                    }`}
                  >
                    <div className={`inline-flex items-center gap-1 ${col.align === "right" ? "justify-end" : ""}`}>
                      <span>{col.header}</span>
                      {canSort && isSorted && (
                        sortOrder === "asc" ? (
                          <ChevronUp className="h-3 w-3 text-[#3b82f6]" />
                        ) : (
                          <ChevronDown className="h-3 w-3 text-[#3b82f6]" />
                        )
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1e2333] text-[#c5cde0]">
            {paginatedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center">
                  <EmptyState title="No Records Found" message={emptyMessage} />
                </td>
              </tr>
            ) : (
              paginatedData.map((item, rowIdx) => (
                <tr
                  key={getItemKey(item, rowIdx)}
                  onClick={() => onRowClick && onRowClick(item)}
                  className={`transition-colors hover:bg-[#1a1e2b] ${
                    onRowClick ? "cursor-pointer" : ""
                  }`}
                >
                  {columns.map((col, colIdx) => (
                    <td
                      key={`cell-${rowIdx}-${colIdx}`}
                      className={`px-4 py-2.5 ${
                        col.align === "right"
                          ? "text-right"
                          : col.align === "center"
                          ? "text-center"
                          : "text-left"
                      }`}
                    >
                      {col.render
                        ? col.render(item, rowIdx)
                        : col.accessor
                        ? col.accessor(item)
                        : col.key
                        ? (item[col.key] as React.ReactNode)
                        : null}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-[#242b3d] px-4 py-2 text-xs text-[#9aa4bf]">
          <span>
            Page {currentPage} of {totalPages} ({sortedData.length} records)
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="rounded p-1 hover:bg-[#1a1e2b] hover:text-[#f0f3fa] disabled:opacity-30 disabled:hover:bg-transparent"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="rounded p-1 hover:bg-[#1a1e2b] hover:text-[#f0f3fa] disabled:opacity-30 disabled:hover:bg-transparent"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DataTable;
