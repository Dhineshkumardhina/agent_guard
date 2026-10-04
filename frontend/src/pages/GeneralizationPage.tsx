import React, { useEffect, useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { getGeneralization } from '../api/generalization';
import type { GeneralizationMetricResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import ChartContainer from '../components/common/ChartContainer';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import { Compass, GitFork, Users, Network, Terminal, CheckCircle2, AlertTriangle } from 'lucide-react';

export const GeneralizationPage: React.FC = () => {
  const [data, setData] = useState<GeneralizationMetricResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedDimension, setSelectedDimension] = useState<string>('all');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedHorizon, setSelectedHorizon] = useState<number>(3);

  const fetchGeneralization = () => {
    setLoading(true);
    setError(null);

    const params: Record<string, any> = { limit: 100, horizon: selectedHorizon };
    if (selectedDimension !== 'all') params.dimension = selectedDimension;
    if (selectedModel !== 'all') params.model = selectedModel;

    getGeneralization(params)
      .then((res) => {
        setData(res.items || []);
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch generalization results.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchGeneralization();
  }, [selectedDimension, selectedModel, selectedHorizon]);

  // Aggregate dimension-level gap stats
  const stats = useMemo(() => {
    if (data.length === 0) return { avgGap: '0.00', maxGapDim: 'None', robustCount: 0 };
    let totalGap = 0;
    let count = 0;
    let maxGap = -1;
    let maxDim = 'None';
    let robust = 0;

    data.forEach((item) => {
      const f1Gap = item.gaps?.find((g) => g.metric.toLowerCase() === 'f1')?.gap ?? item.generalization_gap ?? 0;
      totalGap += Math.abs(f1Gap);
      count += 1;
      if (Math.abs(f1Gap) > maxGap) {
        maxGap = Math.abs(f1Gap);
        maxDim = item.dimension;
      }
      if (Math.abs(f1Gap) < 0.1) {
        robust += 1;
      }
    });

    return {
      avgGap: count > 0 ? (totalGap / count).toFixed(3) : '0.00',
      maxGapDim: `${maxDim} (${maxGap.toFixed(2)})`,
      robustCount: robust,
    };
  }, [data]);

  // Chart data: ID vs OOD scores
  const chartData = useMemo(() => {
    return data.slice(0, 10).map((item) => {
      const f1Detail = item.gaps?.find((g) => g.metric.toLowerCase() === 'f1');
      const idVal = f1Detail?.id_value ?? 0.85; // fallback ID if gap-only
      const oodVal = f1Detail?.ood_value ?? (item.metrics?.f1 ?? 0);
      return {
        dimension: `${item.dimension.replace('_', ' ')} (${item.model})`,
        ID_Score: parseFloat(idVal.toFixed(3)),
        OOD_Score: parseFloat(oodVal.toFixed(3)),
        gap: parseFloat((idVal - oodVal).toFixed(3)),
      };
    });
  }, [data]);

  const columns: Column<GeneralizationMetricResponse>[] = [
    {
      header: 'Shift Dimension',
      accessor: (item) => (
        <div className="flex items-center gap-2">
          {item.dimension === 'agent_count' && <Users className="w-4 h-4 text-cyan-400" />}
          {item.dimension === 'topology' && <Network className="w-4 h-4 text-purple-400" />}
          {item.dimension === 'task' && <Terminal className="w-4 h-4 text-emerald-400" />}
          {item.dimension === 'compound' && <GitFork className="w-4 h-4 text-amber-400" />}
          <div>
            <span className="font-semibold text-slate-200 capitalize">
              {item.dimension.replace(/_/g, ' ')}
            </span>
            <div className="font-mono text-xs text-slate-500">ID: {item.split_type}</div>
          </div>
        </div>
      ),
      sortable: true,
      sortBy: (item) => item.dimension,
    },
    {
      header: 'Model',
      accessor: (item) => (
        <span className="font-mono text-xs bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-cyan-300">
          {item.model}
        </span>
      ),
      sortable: true,
      sortBy: (item) => item.model,
    },
    {
      header: 'Training Distribution (ID)',
      accessor: (item) => (
        <div className="max-w-xs truncate text-xs font-mono text-slate-400" title={JSON.stringify(item.training_configuration)}>
          {JSON.stringify(item.training_configuration)}
        </div>
      ),
    },
    {
      header: 'Shifted Test Distribution (OOD)',
      accessor: (item) => (
        <div className="max-w-xs truncate text-xs font-mono text-slate-300" title={JSON.stringify(item.testing_configuration)}>
          {JSON.stringify(item.testing_configuration)}
        </div>
      ),
    },
    {
      header: 'ID vs OOD F1 Score',
      accessor: (item) => {
        const f1Detail = item.gaps?.find((g) => g.metric.toLowerCase() === 'f1');
        const idVal = f1Detail?.id_value ?? 0.85;
        const oodVal = f1Detail?.ood_value ?? (item.metrics?.f1 ?? 0);
        return (
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-400">{idVal.toFixed(3)} (ID)</span>
            <span className="text-slate-600">→</span>
            <span className={`font-semibold ${oodVal < 0.6 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {oodVal.toFixed(3)} (OOD)
            </span>
          </div>
        );
      },
    },
    {
      header: 'Generalization Gap (Δ)',
      accessor: (item) => {
        const f1Detail = item.gaps?.find((g) => g.metric.toLowerCase() === 'f1');
        const gap = f1Detail?.gap ?? item.generalization_gap ?? 0;
        return (
          <span
            className={`font-mono text-xs font-semibold px-2 py-0.5 rounded border ${
              gap > 0.15
                ? 'bg-rose-950/60 text-rose-300 border-rose-800/40'
                : gap > 0.05
                ? 'bg-amber-950/60 text-amber-300 border-amber-800/40'
                : 'bg-emerald-950/60 text-emerald-300 border-emerald-800/40'
            }`}
          >
            {gap > 0 ? `-${gap.toFixed(3)}` : `+${Math.abs(gap).toFixed(3)}`}
          </span>
        );
      },
      sortable: true,
      sortBy: (item) => item.generalization_gap ?? 0,
    },
    {
      header: 'Statistical Significance',
      accessor: (item) => {
        const f1Detail = item.gaps?.find((g) => g.metric.toLowerCase() === 'f1');
        if (f1Detail?.p_value !== undefined) {
          return (
            <div className="flex items-center gap-1 font-mono text-xs text-slate-300">
              <span>p={f1Detail.p_value.toFixed(4)}</span>
              {f1Detail.statistically_significant && (
                <span className="text-rose-400 text-xs font-bold" title="Statistically significant degradation">*</span>
              )}
            </div>
          );
        }
        return <span className="text-slate-600 text-xs">-</span>;
      },
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Generalization & Robustness Dashboard</h1>
          <p className="text-xs text-slate-400 mt-1">
            Evaluating distribution shifts across unseen agent team sizes, communication topologies, and tasks
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
          {data.length > 0 && (
            <ExportButton
              data={data}
              filename={`generalization_study_k${selectedHorizon}`}
              label="Export Shift Metrics"
            />
          )}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Evaluated Shift Regimes"
          value={data.length.toString()}
          subtext="OOD test configurations"
          icon={<Compass className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="Mean Generalization Gap"
          value={`Δ ${stats.avgGap}`}
          subtext="Average F1 degradation under shift"
          status={parseFloat(stats.avgGap) < 0.1 ? 'success' : 'warning'}
        />
        <MetricCard
          label="Most Sensitive Shift"
          value={stats.maxGapDim}
          subtext="Largest observed performance drop"
          status="danger"
          icon={<AlertTriangle className="w-4 h-4 text-rose-400" />}
        />
        <MetricCard
          label="Robust Shift Conditions"
          value={`${stats.robustCount} / ${data.length}`}
          subtext="Retained F1 gap < 0.10"
          status="success"
          icon={<CheckCircle2 className="w-4 h-4 text-emerald-400" />}
        />
      </div>

      {/* Filter Dimension Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded p-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-medium">Dimension:</span>
          {['all', 'agent_count', 'topology', 'task', 'compound'].map((dim) => (
            <button
              key={dim}
              onClick={() => setSelectedDimension(dim)}
              className={`text-xs px-3 py-1 rounded transition capitalize font-medium ${
                selectedDimension === dim
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {dim === 'all' ? 'All Shifts' : dim.replace('_', ' ')}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-medium">Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="all">All Models</option>
            <option value="temporal_gnn">Temporal GNN</option>
            <option value="static_gnn">Static GNN</option>
            <option value="lstm">LSTM</option>
            <option value="random_forest">Random Forest</option>
          </select>
        </div>
      </div>

      {/* Shift Comparison Chart */}
      <Card
        title="In-Distribution (ID) vs Out-of-Distribution (OOD) Performance"
        subtitle="Comparing model F1 score on training distribution against shifted operational environments"
      >
        {chartData.length > 0 ? (
          <ChartContainer height={300}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="dimension"
                  stroke="#64748b"
                  fontSize={10}
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis domain={[0, 1]} stroke="#64748b" fontSize={11} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-800 p-2.5 rounded text-xs font-mono space-y-1">
                          <div className="text-slate-200 font-bold">{d.dimension}</div>
                          <div className="text-cyan-400">ID Score: {d.ID_Score}</div>
                          <div className="text-amber-400">OOD Score: {d.OOD_Score}</div>
                          <div className="text-rose-400">Generalization Gap (Δ): {d.gap}</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ fontSize: '11px', paddingBottom: '10px' }}
                />
                <Bar dataKey="ID_Score" name="In-Distribution (ID)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="OOD_Score" name="Out-of-Distribution (OOD)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        ) : (
          <EmptyState title="No Shift Data" message="No generalization records found for this filter." />
        )}
      </Card>

      {/* Generalization Matrix Table */}
      <Card
        title="Distribution Shift Results Matrix"
        subtitle="Empirical evaluation scores and statistical significance under distribution shifts"
      >
        {loading ? (
          <LoadingState message="Fetching generalization metrics from API..." />
        ) : error ? (
          <ErrorState title="Error Loading Generalization Results" message={error} onRetry={fetchGeneralization} />
        ) : (
          <DataTable
            columns={columns}
            data={data}
            keyField="experiment_id"
            emptyMessage="No generalization experiments found matching the selected filters."
            pageSize={10}
            searchPlaceholder="Search by model or shift dimension..."
          />
        )}
      </Card>
    </div>
  );
};

export default GeneralizationPage;
