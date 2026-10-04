import React, { useState, useMemo } from "react";
import type { RunGraphResponse, GraphSnapshotSchema } from "../../types";
import { NodeInspector } from "./NodeInspector";
import { EdgeInspector } from "./EdgeInspector";
import { Clock, SkipBack, SkipForward } from "lucide-react";

interface GraphViewerProps {
  graphData?: RunGraphResponse;
  graph?: RunGraphResponse;
  maxSteps?: number;
  className?: string;
  onSnapshotChange?: (snapshotIdx: number) => void;
}

export const GraphViewer: React.FC<GraphViewerProps> = ({
  graphData,
  graph,
  maxSteps: _maxSteps,
  className = "",
  onSnapshotChange,
}) => {
  const effectiveGraph = graphData || graph || { nodes: [], edges: [], temporal_snapshots: [], density: 0, diameter: 0, average_clustering: 0, is_connected: true, run_id: "" };
  const [selectedSnapshotIdx, setSelectedSnapshotIdx] = useState<number>(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeKey, setSelectedEdgeKey] = useState<string | null>(null);

  const snapshots = effectiveGraph.temporal_snapshots || [];
  const currentSnapshot: GraphSnapshotSchema | undefined =
    snapshots[selectedSnapshotIdx] || snapshots[0];

  // Get nodes and edges for the currently selected snapshot (or fallback to full graph)
  const currentNodes = useMemo(() => {
    if (currentSnapshot && currentSnapshot.nodes && currentSnapshot.nodes.length > 0) {
      return currentSnapshot.nodes as Array<{ id: string; role?: string; [key: string]: any }>;
    }
    return (effectiveGraph.nodes || []) as Array<{ id: string; role?: string; [key: string]: any }>;
  }, [currentSnapshot, effectiveGraph.nodes]);

  const currentEdges = useMemo(() => {
    if (currentSnapshot && currentSnapshot.edges && currentSnapshot.edges.length > 0) {
      return currentSnapshot.edges as Array<{ source: string; target: string; [key: string]: any }>;
    }
    return (effectiveGraph.edges || []) as Array<{ source: string; target: string; [key: string]: any }>;
  }, [currentSnapshot, effectiveGraph.edges]);

  // Compute circular layout for nodes (width=600, height=420, center=(300, 210), radius=150)
  const nodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number }> = {};
    const n = currentNodes.length || 1;
    const cx = 300;
    const cy = 210;
    const radius = 150;

    currentNodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / n - Math.PI / 2;
      positions[String(node.id)] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    });

    return positions;
  }, [currentNodes]);

  // Find currently selected node / edge objects
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return currentNodes.find((n) => String(n.id) === selectedNodeId) || null;
  }, [selectedNodeId, currentNodes]);

  const selectedEdge = useMemo(() => {
    if (!selectedEdgeKey) return null;
    const [s, t] = selectedEdgeKey.split("->");
    return (
      currentEdges.find(
        (e) => String(e.source) === s && String(e.target) === t
      ) || null
    );
  }, [selectedEdgeKey, currentEdges]);

  const handlePrevSnapshot = () => {
    if (selectedSnapshotIdx > 0) {
      const nextIdx = selectedSnapshotIdx - 1;
      setSelectedSnapshotIdx(nextIdx);
      if (onSnapshotChange) onSnapshotChange(nextIdx);
    }
  };

  const handleNextSnapshot = () => {
    if (selectedSnapshotIdx < snapshots.length - 1) {
      const nextIdx = selectedSnapshotIdx + 1;
      setSelectedSnapshotIdx(nextIdx);
      if (onSnapshotChange) onSnapshotChange(nextIdx);
    }
  };

  return (
    <div
      className={`flex flex-col rounded-lg border border-[#242b3d] bg-[#141721] overflow-hidden ${className}`}
    >
      {/* Graph Controller / Scrubber Bar */}
      <div className="flex flex-wrap items-center justify-between border-b border-[#242b3d] px-4 py-2.5 text-xs text-[#9aa4bf]">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-[#f0f3fa]">
            Temporal Graph Snapshot:
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={handlePrevSnapshot}
              disabled={selectedSnapshotIdx <= 0}
              className="rounded p-1 hover:bg-[#1a1e2b] hover:text-[#f0f3fa] disabled:opacity-30"
              title="Previous Snapshot"
            >
              <SkipBack className="h-3.5 w-3.5" />
            </button>
            <span className="font-mono text-xs px-2 py-0.5 rounded bg-[#0c0e14] text-[#60a5fa] border border-[#242b3d]">
              Snapshot {selectedSnapshotIdx + 1} / {Math.max(1, snapshots.length)}
            </span>
            <button
              onClick={handleNextSnapshot}
              disabled={selectedSnapshotIdx >= snapshots.length - 1}
              className="rounded p-1 hover:bg-[#1a1e2b] hover:text-[#f0f3fa] disabled:opacity-30"
              title="Next Snapshot"
            >
              <SkipForward className="h-3.5 w-3.5" />
            </button>
          </div>

          {currentSnapshot && (
            <div className="flex items-center gap-1.5 text-[#5e6984] font-mono text-[11px]">
              <Clock className="h-3 w-3" />
              <span>Step: {currentSnapshot.step_idx}</span>
              <span>·</span>
              <span>t={currentSnapshot.timestamp.toFixed(2)}s</span>
            </div>
          )}
        </div>

        {/* Global Topology Summary */}
        <div className="flex items-center gap-4 text-[11px]">
          <span>
            <strong className="text-[#f0f3fa] font-mono">{currentNodes.length}</strong> Agents
          </span>
          <span>
            <strong className="text-[#f0f3fa] font-mono">{currentEdges.length}</strong> Edges
          </span>
          <span>
            Density:{" "}
            <strong className="text-[#f0f3fa] font-mono">
              {(effectiveGraph as any).density ? (effectiveGraph as any).density.toFixed(3) : "N/A"}
            </strong>
          </span>
        </div>
      </div>

      {/* Snapshot Slider Range if multiple snapshots exist */}
      {snapshots.length > 1 && (
        <div className="border-b border-[#242b3d] bg-[#0c0e14] px-4 py-2 flex items-center gap-3">
          <span className="text-[11px] text-[#5e6984] font-mono">Scrubber:</span>
          <input
            type="range"
            min={0}
            max={snapshots.length - 1}
            value={selectedSnapshotIdx}
            onChange={(e) => {
              const idx = parseInt(e.target.value, 10);
              setSelectedSnapshotIdx(idx);
              if (onSnapshotChange) onSnapshotChange(idx);
            }}
            className="flex-1 accent-[#3b82f6] cursor-pointer"
          />
        </div>
      )}

      {/* Main Canvas Area */}
      <div className="relative flex flex-col md:flex-row min-h-[420px]">
        {/* SVG Drawing Canvas */}
        <div className="relative flex-1 bg-[#0b0d13] p-4 flex items-center justify-center">
          <svg
            viewBox="0 0 600 420"
            className="w-full h-full max-h-[500px]"
            onClick={() => {
              setSelectedNodeId(null);
              setSelectedEdgeKey(null);
            }}
          >
            {/* Defs for directional arrows */}
            <defs>
              <marker
                id="arrow-normal"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
              </marker>
              <marker
                id="arrow-selected"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#60a5fa" />
              </marker>
              <marker
                id="arrow-contradiction"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#ef4444" />
              </marker>
            </defs>

            {/* Render Edges */}
            {currentEdges.map((edge) => {
              const srcPos = nodePositions[String(edge.source)];
              const tgtPos = nodePositions[String(edge.target)];
              if (!srcPos || !tgtPos) return null;

              const edgeKey = `${edge.source}->${edge.target}`;
              const isSelected = selectedEdgeKey === edgeKey;
              const hasContradiction =
                Boolean(edge.contradiction_score && edge.contradiction_score > 0.2);

              let strokeColor = "#334155";
              let marker = "url(#arrow-normal)";

              if (hasContradiction) {
                strokeColor = "#ef4444";
                marker = "url(#arrow-contradiction)";
              } else if (isSelected) {
                strokeColor = "#60a5fa";
                marker = "url(#arrow-selected)";
              }

              return (
                <g key={edgeKey} className="cursor-pointer group">
                  <line
                    x1={srcPos.x}
                    y1={srcPos.y}
                    x2={tgtPos.x}
                    y2={tgtPos.y}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? 3 : hasContradiction ? 2.5 : 1.5}
                    strokeDasharray={hasContradiction ? "4 2" : "none"}
                    markerEnd={marker}
                    className="transition-colors group-hover:stroke-[#93c5fd]"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedEdgeKey(edgeKey);
                      setSelectedNodeId(null);
                    }}
                  />
                  {/* Invisible thicker line for easier clicking */}
                  <line
                    x1={srcPos.x}
                    y1={srcPos.y}
                    x2={tgtPos.x}
                    y2={tgtPos.y}
                    stroke="transparent"
                    strokeWidth={14}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedEdgeKey(edgeKey);
                      setSelectedNodeId(null);
                    }}
                  />
                </g>
              );
            })}

            {/* Render Nodes */}
            {currentNodes.map((node) => {
              const pos = nodePositions[String(node.id)];
              if (!pos) return null;

              const isSelected = selectedNodeId === String(node.id);
              const isFailed = Boolean(
                node.failure_manifested ||
                node.status === "failed" ||
                (node.error_count && node.error_count > 0)
              );

              const nodeColor = isFailed
                ? "#7f1d1d"
                : isSelected
                ? "#1d4ed8"
                : "#1b2030";
              const borderColor = isFailed
                ? "#b91c1c"
                : isSelected
                ? "#60a5fa"
                : "#333d56";

              return (
                <g
                  key={String(node.id)}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer transition-transform hover:scale-105"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedNodeId(String(node.id));
                    setSelectedEdgeKey(null);
                  }}
                >
                  {/* Outer pulse circle if failed */}
                  {isFailed && (
                    <circle
                      r={24}
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                      className="animate-spin"
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    r={18}
                    fill={nodeColor}
                    stroke={borderColor}
                    strokeWidth={isSelected ? 3 : 2}
                  />

                  {/* Node Label Initial */}
                  <text
                    textAnchor="middle"
                    dy="5"
                    className="font-mono text-[11px] font-bold fill-[#f0f3fa] pointer-events-none select-none"
                  >
                    {String(node.id || "A").substring(0, 2).toUpperCase()}
                  </text>

                  {/* Agent Full ID below node */}
                  <text
                    textAnchor="middle"
                    dy="32"
                    className="font-mono text-[10px] fill-[#9aa4bf] pointer-events-none select-none"
                  >
                    {String(node.id)}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Quick Legend at bottom of canvas */}
          <div className="absolute bottom-2 left-3 flex items-center gap-3 text-[11px] text-[#5e6984]">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#1b2030] border border-[#333d56]" />
              Active
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#ef4444]" />
              Failure Manifested
            </span>
            <span className="flex items-center gap-1">
              <span className="h-0.5 w-3 bg-[#475569]" />
              Interaction
            </span>
            <span className="flex items-center gap-1">
              <span className="h-0.5 w-3 bg-[#ef4444]" />
              Contradiction &gt; 20%
            </span>
          </div>
        </div>

        {/* Selected Inspector Drawer / Panel */}
        {(selectedNode || selectedEdge) && (
          <div className="w-full border-t border-[#242b3d] p-3 md:w-80 md:border-l md:border-t-0">
            {selectedNode && (
              <NodeInspector
                node={selectedNode}
                onClose={() => setSelectedNodeId(null)}
              />
            )}
            {selectedEdge && (
              <EdgeInspector
                edge={selectedEdge}
                onClose={() => setSelectedEdgeKey(null)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphViewer;
