import React, { useEffect, useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { getAblations } from '../api/ablations';
import type { AblationResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import ChartContainer from '../components/common/ChartContainer';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import { Scissors, Info, Layers } from 'lucide-react';

export const AblationsPage: React.FC = () => {
  const [ablations, setAblations] = useState<AblationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<number>(3);

  const fetchAblations = () => {
    setLoading(true);
    setError(null);
    getAblations({ horizon: selectedHorizon, limit: 100 })
      .then((res) => {
        setAblations(res.items || []);
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch ablation study results.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchAblations();
  }, [selectedHorizon]);

  // Identify baseline (Full Model)
  const fullModel = useMemo(() => {
    return (
      ablations.find(
        (a) =>
          a.removed_component.toLowerCase().includes('none') ||
          a.ablation_name.toLowerCase().includes('full') ||
          a.removed_component === 'None'
      ) || ablations[0]
    );
  }, [ablations]);

  // Compute delta metrics relative to full model
  const tableData: Array<AblationResponse & { deltaF1?: number; deltaAuprc?: number; deltaLeadTime?: number }> = useMemo(() => {
    if (!fullModel) return ablations;
    return ablations.map((a) => {
      const deltaF1 = a.f1 - fullModel.f1;
      const deltaAuprc = a.auprc - fullModel.auprc;
      const deltaLeadTime = a.mean_lead_time - fullModel.mean_lead_time;
      return {
        ...a,
        deltaF1,
        deltaAuprc,
        deltaLeadTime,
      };
    });
  }, [ablations, fullModel]);

  // Data for Charts
  const chartData = useMemo(() => {
    return tableData.map((item) => ({
      name: item.removed_component === 'None' ? 'Full Model' : item.removed_component,
      f1: parseFloat(item.f1.toFixed(3)),
      auprc: parseFloat(item.auprc.toFixed(3)),
      leadTime: parseFloat(item.mean_lead_time.toFixed(2)),
      deltaF1: parseFloat((item.deltaF1 ?? 0).toFixed(3)),
      isBaseline: item.removed_component === 'None' || item.ablation_name.toLowerCase().includes('full'),
    }));
  }, [tableData]);

  const columns: Column<any>[] = [
    {
      header: 'Component Removed',
      accessor: (a) => (
        <div>
          <span className="font-semibold text-slate-200">
            {a.removed_component === 'None' ? 'None (Full Temporal GNN)' : a.removed_component}
          </span>
          <div className="font-mono text-xs text-slate-500">{a.ablation_name}</div>
        </div>
      ),
      sortable: true,
      sortBy: (a) => a.removed_component,
    },
    {
      header: 'F1 Score',
      accessor: (a) => (
        <div className="flex items-center gap-1.5 font-mono text-xs">
          <span className="text-slate-200 font-semibold">{a.f1.toFixed(3)}</span>
          {a.deltaF1 !== undefined && a.removed_component !== 'None' && (
            <span
              className={`text-xs ${
                a.deltaF1 < -0.05 ? 'text-rose-400' : a.deltaF1 < 0 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              ({a.deltaF1 > 0 ? `+${a.deltaF1.toFixed(3)}` : a.deltaF1.toFixed(3)})
            </span>
          )}
        </div>
      ),
      sortable: true,
      sortBy: (a) => a.f1,
    },
    {
      header: 'AUPRC',
      accessor: (a) => (
        <div className="flex items-center gap-1.5 font-mono text-xs">
          <span className="text-slate-200 font-semibold">{a.auprc.toFixed(3)}</span>
          {a.deltaAuprc !== undefined && a.removed_component !== 'None' && (
            <span
              className={`text-xs ${
                a.deltaAuprc < -0.05 ? 'text-rose-400' : a.deltaAuprc < 0 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              ({a.deltaAuprc > 0 ? `+${a.deltaAuprc.toFixed(3)}` : a.deltaAuprc.toFixed(3)})
            </span>
          )}
        </div>
      ),
      sortable: true,
      sortBy: (a) => a.auprc,
    },
    {
      header: 'AUROC',
      accessor: (a) => <span className="font-mono text-xs text-slate-300">{a.auroc.toFixed(3)}</span>,
      sortable: true,
      sortBy: (a) => a.auroc,
    },
    {
      header: 'Mean Lead Time',
      accessor: (a) => (
        <div className="flex items-center gap-1.5 font-mono text-xs">
          <span className="text-cyan-300">{a.mean_lead_time.toFixed(1)} steps</span>
          {a.deltaLeadTime !== undefined && a.removed_component !== 'None' && (
            <span
              className={`text-xs ${
                a.deltaLeadTime < -1 ? 'text-rose-400' : a.deltaLeadTime < 0 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              ({a.deltaLeadTime > 0 ? `+${a.deltaLeadTime.toFixed(1)}` : a.deltaLeadTime.toFixed(1)})
            </span>
          )}
        </div>
      ),
      sortable: true,
      sortBy: (a) => a.mean_lead_time,
    },
    {
      header: 'False Alarm Rate',
      accessor: (a) => (
        <span className="font-mono text-xs text-slate-300">
          {(a.false_alarm_rate * 100).toFixed(1)}%
        </span>
      ),
      sortable: true,
      sortBy: (a) => a.false_alarm_rate,
    },
    {
      header: 'Recall',
      accessor: (a) => <span className="font-mono text-xs text-slate-300">{a.recall.toFixed(3)}</span>,
    },
    {
      header: 'Precision',
      accessor: (a) => <span className="font-mono text-xs text-slate-300">{a.precision.toFixed(3)}</span>,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Ablation Study Dashboard</h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical component attribution: evaluating predictive performance degradation under component removal
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded px-3 py-1.5">
            <span className="text-xs text-slate-400 font-medium">Horizon:</span>
            <select
              value={selectedHorizon}
              onChange={(e) => setSelectedHorizon(parseInt(e.target.value, 10))}
              className="bg-transparent text-xs text-cyan-400 font-mono focus:outline-none cursor-pointer"
            >
              <option value={1} className="bg-slate-950 text-slate-200">k = 1</option>
              <option value={2} className="bg-slate-950 text-slate-200">k = 2</option>
              <option value={3} className="bg-slate-950 text-slate-200">k = 3</option>
              <option value={5} className="bg-slate-950 text-slate-200">k = 5</option>
            </select>
          </div>
          {ablations.length > 0 && (
            <ExportButton
              data={tableData}
              filename={`ablation_study_k${selectedHorizon}`}
              label="Export Ablation Data"
            />
          )}
        </div>
      </div>

      {/* Non-Causality Scientific Disclaimer */}
      <div className="p-3.5 rounded bg-slate-900/90 border border-amber-500/30 text-xs text-slate-300 flex items-start gap-3">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-amber-300">Methodological Note on Component Attribution:</strong> Components are
          strictly labeled <span className="font-semibold text-slate-100">"Component Removed"</span>. Ablation studies
          measure the empirical drop in model predictive performance when an architectural or informational channel is
          ablated. They do not demonstrate causal necessity or isolated mechanistic causation in open multi-agent
          environments.
        </div>
      </div>

      {/* Baseline KPI Summary */}
      {fullModel && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Full Model Baseline F1"
            value={fullModel.f1.toFixed(3)}
            subtext={`Horizon k=${selectedHorizon} · Seed ${fullModel.seed}`}
            status="success"
            icon={<Layers className="w-4 h-4 text-cyan-400" />}
          />
          <MetricCard
            label="Full Model AUPRC"
            value={fullModel.auprc.toFixed(3)}
            status="success"
          />
          <MetricCard
            label="Full Model Mean Lead Time"
            value={`${fullModel.mean_lead_time.toFixed(1)} steps`}
            status="info"
          />
          <MetricCard
            label="Ablated Configurations"
            value={ablations.length.toString()}
            subtext="Architectural & feature variants"
            icon={<Scissors className="w-4 h-4 text-purple-400" />}
          />
        </div>
      )}

      {/* Ablation Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart 1: Component Removed vs F1 */}
        <Card
          title="Component Removed vs F1 Score"
          subtitle="Impact of removing components on failure prediction F1"
        >
          {chartData.length > 0 ? (
            <ChartContainer height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 70, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={10} width={90} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-slate-900 border border-slate-800 p-2 rounded text-xs font-mono">
                            <div className="text-slate-300 font-bold">{d.name}</div>
                            <div className="text-cyan-400">F1: {d.f1}</div>
                            {!d.isBaseline && (
                              <div className="text-rose-400">Δ F1: {d.deltaF1 > 0 ? `+${d.deltaF1}` : d.deltaF1}</div>
                            )}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  {fullModel && (
                    <ReferenceLine x={fullModel.f1} stroke="#06b6d4" strokeDasharray="3 3" strokeWidth={1.5} />
                  )}
                  <Bar dataKey="f1" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isBaseline ? '#06b6d4' : entry.deltaF1 < -0.1 ? '#f43f5e' : '#64748b'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <EmptyState title="No Chart Data" message="No ablation items recorded for this horizon." />
          )}
        </Card>

        {/* Chart 2: Component Removed vs AUPRC */}
        <Card
          title="Component Removed vs AUPRC"
          subtitle="Impact on Area Under the Precision-Recall Curve"
        >
          {chartData.length > 0 ? (
            <ChartContainer height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 70, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={10} width={90} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-slate-900 border border-slate-800 p-2 rounded text-xs font-mono">
                            <div className="text-slate-300 font-bold">{d.name}</div>
                            <div className="text-purple-400">AUPRC: {d.auprc}</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  {fullModel && (
                    <ReferenceLine x={fullModel.auprc} stroke="#a855f7" strokeDasharray="3 3" strokeWidth={1.5} />
                  )}
                  <Bar dataKey="auprc" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-auprc-${index}`} fill={entry.isBaseline ? '#a855f7' : '#475569'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <EmptyState title="No Chart Data" message="No ablation items recorded for this horizon." />
          )}
        </Card>

        {/* Chart 3: Component Removed vs Lead Time */}
        <Card
          title="Component Removed vs Lead Time"
          subtitle="Impact on mean steps of advance early warning"
        >
          {chartData.length > 0 ? (
            <ChartContainer height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 70, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#64748b" fontSize={10} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={10} width={90} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-slate-900 border border-slate-800 p-2 rounded text-xs font-mono">
                            <div className="text-slate-300 font-bold">{d.name}</div>
                            <div className="text-emerald-400">Lead Time: {d.leadTime} steps</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  {fullModel && (
                    <ReferenceLine x={fullModel.mean_lead_time} stroke="#10b981" strokeDasharray="3 3" strokeWidth={1.5} />
                  )}
                  <Bar dataKey="leadTime" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-lt-${index}`} fill={entry.isBaseline ? '#10b981' : '#334155'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <EmptyState title="No Chart Data" message="No ablation items recorded for this horizon." />
          )}
        </Card>
      </div>

      {/* Detailed Ablations Table */}
      <Card
        title="Component Removal Performance Matrix"
        subtitle="Empirical evaluation metrics and differential gaps (Δ) compared to the unablated Temporal GNN"
      >
        {loading ? (
          <LoadingState message="Fetching ablation experiments from API..." />
        ) : error ? (
          <ErrorState title="Error Loading Ablations" message={error} onRetry={fetchAblations} />
        ) : (
          <DataTable
            columns={columns}
            data={tableData}
            keyField="experiment_id"
            emptyMessage="No ablation experiments available for the selected horizon."
            pageSize={10}
            searchPlaceholder="Search by removed component or experiment name..."
          />
        )}
      </Card>
    </div>
  );
};

export default AblationsPage;
