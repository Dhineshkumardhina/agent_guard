import React, { useEffect, useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { getRuns } from '../api/runs';
import { getPredictions } from '../api/predictions';
import type { RunResponse, PredictionResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { RiskBadge } from '../components/common/Badges';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import ChartContainer from '../components/common/ChartContainer';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import { Clock, ShieldAlert, CheckCircle, XCircle } from 'lucide-react';

export const EarlyWarningPage: React.FC = () => {
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('temporal_gnn');
  const [selectedHorizon, setSelectedHorizon] = useState<number>(3);
  const [threshold, setThreshold] = useState<number>(0.5);

  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingPreds, setLoadingPreds] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Fetch available runs
  useEffect(() => {
    setLoadingRuns(true);
    getRuns({ limit: 50 })
      .then((res) => {
        const items = res.items || [];
        setRuns(items);
        if (items.length > 0) {
          const cascadedRun = items.find(
            (r) =>
              (r.cascading_failure_step ?? r.cascade_step) !== undefined &&
              (r.cascading_failure_step ?? r.cascade_step) !== null
          );
          const firstRun = cascadedRun || items[0];
          setSelectedRunId(firstRun.id || firstRun.run_id || '');
        }
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch runs.');
      })
      .finally(() => {
        setLoadingRuns(false);
      });
  }, []);

  // 2. Fetch predictions for selected run & model
  useEffect(() => {
    if (!selectedRunId) return;
    setLoadingPreds(true);
    getPredictions({
      run_id: selectedRunId,
      model: selectedModel || undefined,
      horizon: selectedHorizon,
      limit: 100,
    })
      .then((res) => {
        const items = res.items || [];
        setPredictions(items);
        if (items.length > 0 && items[0].threshold !== undefined) {
          setThreshold(items[0].threshold);
        }
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch predictions.');
      })
      .finally(() => {
        setLoadingPreds(false);
      });
  }, [selectedRunId, selectedModel, selectedHorizon]);

  // Selected run metadata
  const currentRun = useMemo(() => {
    return runs.find((r) => r.id === selectedRunId || r.run_id === selectedRunId) || null;
  }, [runs, selectedRunId]);

  // Lead time and warning calculations
  const warningMetrics = useMemo(() => {
    const sorted = [...predictions].sort((a, b) => (a.step_idx ?? a.timestamp) - (b.step_idx ?? b.timestamp));
    const firstWarning = sorted.find((p) => p.predicted_probability >= threshold);
    const warningStep = firstWarning?.step_idx ?? null;
    const cascadeStep = currentRun?.cascade_step ?? null;

    let leadTime: number | null = null;
    let status: 'successful_warning' | 'false_alarm' | 'missed_failure' | 'true_negative' = 'true_negative';

    if (cascadeStep !== null) {
      if (warningStep !== null && warningStep <= cascadeStep) {
        leadTime = cascadeStep - warningStep;
        status = 'successful_warning';
      } else {
        status = 'missed_failure';
      }
    } else {
      if (warningStep !== null) {
        status = 'false_alarm';
      } else {
        status = 'true_negative';
      }
    }

    return {
      warningStep,
      cascadeStep,
      leadTime,
      status,
      firstWarning,
    };
  }, [predictions, threshold, currentRun]);

  // Chart data preparation
  const chartData = useMemo(() => {
    return [...predictions]
      .sort((a, b) => (a.step_idx ?? a.timestamp) - (b.step_idx ?? b.timestamp))
      .map((p) => ({
        step: p.step_idx ?? p.timestamp,
        probability: p.predicted_probability,
        threshold: threshold,
        isWarning: p.predicted_probability >= threshold,
        risk: p.risk_level || (p.predicted_probability >= threshold ? 'HIGH' : 'LOW'),
      }));
  }, [predictions, threshold]);

  const columns: Column<PredictionResponse>[] = [
    {
      header: 'Step',
      accessor: (p) => <span className="font-mono text-xs text-slate-300">{p.step_idx ?? '-'}</span>,
      sortable: true,
      sortBy: (p) => p.step_idx ?? 0,
    },
    {
      header: 'Timestamp',
      accessor: (p) => <span className="font-mono text-xs text-slate-400">{p.timestamp.toFixed(2)}s</span>,
    },
    {
      header: 'Predicted Probability',
      accessor: (p) => (
        <span
          className={`font-mono text-xs font-semibold ${
            p.predicted_probability >= threshold ? 'text-rose-400' : 'text-emerald-400'
          }`}
        >
          {p.predicted_probability.toFixed(4)}
        </span>
      ),
      sortable: true,
      sortBy: (p) => p.predicted_probability,
    },
    {
      header: 'Threshold',
      accessor: () => <span className="font-mono text-xs text-slate-400">{threshold.toFixed(2)}</span>,
    },
    {
      header: 'Model Decision',
      accessor: (p) =>
        p.predicted_probability >= threshold ? (
          <span className="text-xs px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40">
            Warning (1)
          </span>
        ) : (
          <span className="text-xs px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
            Normal (0)
          </span>
        ),
    },
    {
      header: 'Risk Level',
      accessor: (p) => <RiskBadge risk={p.risk_level || (p.predicted_probability >= threshold ? 'HIGH' : 'LOW')} />,
    },
    {
      header: 'Lead Time',
      accessor: (p) =>
        p.lead_time !== undefined && p.lead_time !== null ? (
          <span className="font-mono text-xs text-cyan-300 font-semibold">{p.lead_time} steps</span>
        ) : (
          <span className="text-slate-600 text-xs">-</span>
        ),
    },
  ];

  if (loadingRuns) {
    return <LoadingState message="Loading simulation runs for early-warning evaluation..." />;
  }

  if (error && runs.length === 0) {
    return <ErrorState title="Failed to Load Early Warning Data" message={error} onRetry={() => window.location.reload()} />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Early Warning & Lead Time Analysis</h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical evaluation of failure warning lead times, probability evolution, and detection fidelity
          </p>
        </div>
        {predictions.length > 0 && (
          <ExportButton
            data={predictions}
            filename={`early_warning_${selectedRunId}_${selectedModel}_k${selectedHorizon}`}
            label="Export Warnings Data"
          />
        )}
      </div>

      {/* Control Bar: Select Run, Model, Horizon */}
      <div className="bg-slate-900 border border-slate-800 rounded p-4 flex flex-wrap items-center gap-4">
        <div className="flex-1 min-w-[220px]">
          <label className="block text-xs font-medium text-slate-400 mb-1">Simulation Run</label>
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            {runs.map((r) => {
              const rId = r.id || r.run_id || '';
              const cascadeStep = r.cascading_failure_step ?? r.cascade_step;
              return (
                <option key={rId} value={rId}>
                  {rId.slice(0, 14)}... ({r.task} · {r.topology} · {r.status}
                  {cascadeStep !== undefined && cascadeStep !== null ? ` · cascade @ ${cascadeStep}` : ''})
                </option>
              );
            })}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Prediction Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="temporal_gnn">Temporal GNN (TGN)</option>
            <option value="static_gnn">Static GNN (GAT)</option>
            <option value="lstm">LSTM</option>
            <option value="gru">GRU</option>
            <option value="random_forest">Random Forest</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="rule_based">Rule-Based</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Prediction Horizon (k)</label>
          <select
            value={selectedHorizon}
            onChange={(e) => setSelectedHorizon(parseInt(e.target.value, 10))}
            className="bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value={1}>k = 1 step</option>
            <option value={2}>k = 2 steps</option>
            <option value={3}>k = 3 steps</option>
            <option value={5}>k = 5 steps</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Decision Threshold (τ)</label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-24 accent-cyan-500 cursor-pointer"
            />
            <span className="font-mono text-xs text-slate-300 w-8">{threshold.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Prediction Horizon"
          value={`k = ${selectedHorizon}`}
          subtext="Lookahead window"
          icon={<Clock className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="Decision Threshold"
          value={`τ = ${threshold.toFixed(2)}`}
          subtext="Warning trigger point"
        />
        <MetricCard
          label="Warning Time"
          value={warningMetrics.warningStep !== null ? `Step ${warningMetrics.warningStep}` : 'None'}
          subtext={
            warningMetrics.firstWarning
              ? `P = ${warningMetrics.firstWarning.predicted_probability.toFixed(3)}`
              : 'Never exceeded threshold'
          }
          status={warningMetrics.warningStep !== null ? 'warning' : 'neutral'}
          icon={<ShieldAlert className="w-4 h-4 text-amber-400" />}
        />
        <MetricCard
          label="System Cascade Onset"
          value={warningMetrics.cascadeStep !== null ? `Step ${warningMetrics.cascadeStep}` : 'No Cascade'}
          subtext={currentRun?.status === 'cascaded' ? 'Actual systemic failure' : 'Trajectory completed normally'}
          status={warningMetrics.cascadeStep !== null ? 'danger' : 'success'}
        />
        <MetricCard
          label="Mean Lead Time"
          value={
            warningMetrics.leadTime !== null
              ? `${warningMetrics.leadTime} steps`
              : warningMetrics.cascadeStep !== null
              ? '0 (Missed)'
              : 'N/A'
          }
          subtext={
            warningMetrics.status === 'successful_warning'
              ? 'Advance warning provided'
              : warningMetrics.status === 'false_alarm'
              ? 'False alarm on normal run'
              : warningMetrics.status === 'missed_failure'
              ? 'Failed to warn in advance'
              : 'Clean trajectory'
          }
          status={
            warningMetrics.status === 'successful_warning'
              ? 'success'
              : warningMetrics.status === 'false_alarm'
              ? 'warning'
              : warningMetrics.status === 'missed_failure'
              ? 'danger'
              : 'neutral'
          }
        />
      </div>

      {/* Status Banner */}
      <div
        className={`p-3 rounded border text-xs flex items-center justify-between ${
          warningMetrics.status === 'successful_warning'
            ? 'bg-emerald-950/40 border-emerald-800/40 text-emerald-300'
            : warningMetrics.status === 'false_alarm'
            ? 'bg-amber-950/40 border-amber-800/40 text-amber-300'
            : warningMetrics.status === 'missed_failure'
            ? 'bg-rose-950/40 border-rose-800/40 text-rose-300'
            : 'bg-slate-900 border-slate-800 text-slate-300'
        }`}
      >
        <div className="flex items-center gap-2 font-medium">
          {warningMetrics.status === 'successful_warning' && <CheckCircle className="w-4 h-4 text-emerald-400" />}
          {warningMetrics.status === 'false_alarm' && <ShieldAlert className="w-4 h-4 text-amber-400" />}
          {warningMetrics.status === 'missed_failure' && <XCircle className="w-4 h-4 text-rose-400" />}
          {warningMetrics.status === 'true_negative' && <CheckCircle className="w-4 h-4 text-slate-400" />}
          <span>
            {warningMetrics.status === 'successful_warning' &&
              `SUCCESSFUL WARNING: Model crossed threshold τ=${threshold.toFixed(2)} at Step ${warningMetrics.warningStep}, granting ${warningMetrics.leadTime} steps of lead time prior to system cascade at Step ${warningMetrics.cascadeStep}.`}
            {warningMetrics.status === 'false_alarm' &&
              `FALSE ALARM: Warning raised at Step ${warningMetrics.warningStep}, but this trajectory had no systemic cascade.`}
            {warningMetrics.status === 'missed_failure' &&
              `MISSED DETECTION: Cascade occurred at Step ${warningMetrics.cascadeStep} without model probability exceeding threshold τ=${threshold.toFixed(2)}.`}
            {warningMetrics.status === 'true_negative' &&
              `TRUE NEGATIVE: Trajectory completed normally and model correctly kept risk probability below threshold.`}
          </span>
        </div>
      </div>

      {/* Probability Evolution Curve */}
      <Card
        title="Time → Predicted Failure Probability Curve"
        subtitle={`Empirical probability evolution over simulation steps with decision threshold τ and failure boundary`}
      >
        {loadingPreds ? (
          <LoadingState message="Fetching probability sequence for selected model and run..." />
        ) : chartData.length > 0 ? (
          <ChartContainer height={340}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="step"
                  stroke="#64748b"
                  fontSize={11}
                  label={{ value: 'Simulation Step', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 1]}
                  stroke="#64748b"
                  fontSize={11}
                  label={{ value: 'Predicted Failure Risk P(Y=1)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-800 p-2.5 rounded shadow-xl text-xs font-mono space-y-1">
                          <div className="text-slate-300 font-bold">Step {data.step}</div>
                          <div className="text-cyan-400">Risk P: {data.probability.toFixed(4)}</div>
                          <div className="text-slate-500">Threshold τ: {data.threshold.toFixed(2)}</div>
                          <div className="text-amber-400">Risk Level: {data.risk}</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                {/* Decision Threshold Line */}
                <ReferenceLine
                  y={threshold}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{
                    value: `Threshold τ=${threshold.toFixed(2)}`,
                    fill: '#f59e0b',
                    fontSize: 10,
                    position: 'insideTopRight',
                  }}
                />
                {/* Warning Step Line */}
                {warningMetrics.warningStep !== null && (
                  <ReferenceLine
                    x={warningMetrics.warningStep}
                    stroke="#06b6d4"
                    strokeDasharray="3 3"
                    strokeWidth={1.5}
                    label={{
                      value: `Warning @ Step ${warningMetrics.warningStep}`,
                      fill: '#06b6d4',
                      fontSize: 10,
                      position: 'top',
                    }}
                  />
                )}
                {/* Actual Cascade Line */}
                {warningMetrics.cascadeStep !== null && (
                  <ReferenceLine
                    x={warningMetrics.cascadeStep}
                    stroke="#f43f5e"
                    strokeWidth={2}
                    label={{
                      value: `Cascade @ Step ${warningMetrics.cascadeStep}`,
                      fill: '#f43f5e',
                      fontSize: 10,
                      position: 'top',
                    }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#0284c7' }}
                  activeDot={{ r: 6, fill: '#38bdf8' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        ) : (
          <EmptyState
            title="No Predictions Available"
            message={`No inference records found for Run '${selectedRunId}' using Model '${selectedModel}' at horizon k=${selectedHorizon}.`}
          />
        )}
      </Card>

      {/* Raw Predictions Table */}
      <Card
        title="Prediction Log"
        subtitle="Individual probability estimates per step for the selected trajectory"
      >
        <DataTable
          columns={columns}
          data={predictions}
          keyField="prediction_id"
          emptyMessage="No predictions recorded."
          pageSize={10}
        />
      </Card>
    </div>
  );
};

export default EarlyWarningPage;
