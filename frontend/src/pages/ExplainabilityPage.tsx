import React, { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { getExplanations } from '../api/explainability';
import type { ExplanationResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import ChartContainer from '../components/common/ChartContainer';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import { Sparkles, Brain, ShieldAlert, Users, ExternalLink, Info } from 'lucide-react';

export const ExplainabilityPage: React.FC = () => {
  const [explanations, setExplanations] = useState<ExplanationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [caseFilter, setCaseFilter] = useState<string>('all');

  const fetchExplanations = () => {
    setLoading(true);
    setError(null);
    getExplanations({ limit: 100 })
      .then((res) => {
        setExplanations(res.items || []);
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch model explanations.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchExplanations();
  }, []);

  // Filter by case type
  const filteredExplanations = useMemo(() => {
    if (caseFilter === 'all') return explanations;
    return explanations.filter((e) => e.case_type?.toLowerCase() === caseFilter.toLowerCase());
  }, [explanations, caseFilter]);

  // Aggregate global top features across loaded explanations
  const aggregatedFeatures = useMemo(() => {
    const scores: Record<string, { total: number; count: number }> = {};
    explanations.forEach((exp) => {
      (exp.important_features || []).forEach((feat) => {
        if (!scores[feat.feature_name]) scores[feat.feature_name] = { total: 0, count: 0 };
        scores[feat.feature_name].total += feat.importance_score;
        scores[feat.feature_name].count += 1;
      });
    });

    return Object.entries(scores)
      .map(([name, val]) => ({
        name,
        score: parseFloat((val.total / Math.max(1, val.count)).toFixed(3)),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }, [explanations]);

  // Aggregate top agent contributions
  const aggregatedAgents = useMemo(() => {
    const scores: Record<string, { total: number; count: number; role?: string }> = {};
    explanations.forEach((exp) => {
      (exp.important_agents || []).forEach((ag) => {
        if (!scores[ag.agent_id]) scores[ag.agent_id] = { total: 0, count: 0, role: ag.role };
        scores[ag.agent_id].total += ag.importance_score;
        scores[ag.agent_id].count += 1;
      });
    });

    return Object.entries(scores)
      .map(([agent_id, val]) => ({
        agent_id,
        role: val.role || 'Agent',
        score: parseFloat((val.total / Math.max(1, val.count)).toFixed(3)),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }, [explanations]);

  const columns: Column<ExplanationResponse>[] = [
    {
      header: 'Case / Explanation ID',
      accessor: (e) => (
        <Link
          to={`/explainability/${e.explanation_id}`}
          className="font-mono text-xs font-semibold text-cyan-400 hover:text-cyan-300 hover:underline flex items-center gap-1.5"
        >
          {e.explanation_id.slice(0, 16)}...
          <ExternalLink className="w-3 h-3 opacity-60" />
        </Link>
      ),
      sortable: true,
      sortBy: (e) => e.explanation_id,
    },
    {
      header: 'Case Type',
      accessor: (e) => {
        const ct = e.case_type || 'unclassified';
        const color =
          ct.includes('true_positive') || ct.includes('tp')
            ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/40'
            : ct.includes('false_positive') || ct.includes('fp')
            ? 'bg-amber-950/60 text-amber-300 border-amber-800/40'
            : ct.includes('false_negative') || ct.includes('fn')
            ? 'bg-rose-950/60 text-rose-300 border-rose-800/40'
            : 'bg-slate-900 text-slate-300 border-slate-800';
        return (
          <span className={`text-xs px-2 py-0.5 rounded border uppercase font-mono font-semibold ${color}`}>
            {ct.replace(/_/g, ' ')}
          </span>
        );
      },
      sortable: true,
      sortBy: (e) => e.case_type || '',
    },
    {
      header: 'Run ID',
      accessor: (e) => (
        <Link
          to={`/runs/${e.run_id}`}
          className="font-mono text-xs text-slate-300 hover:text-cyan-400 hover:underline"
        >
          {e.run_id.slice(0, 12)}
        </Link>
      ),
    },
    {
      header: 'Risk Probability',
      accessor: (e) => (
        <div className="flex items-center gap-2">
          <div className="w-14 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${
                e.predicted_probability >= e.threshold ? 'bg-rose-500' : 'bg-emerald-400'
              }`}
              style={{ width: `${Math.round(e.predicted_probability * 100)}%` }}
            />
          </div>
          <span className="font-mono text-xs font-semibold text-slate-200">
            {e.predicted_probability.toFixed(3)}
          </span>
        </div>
      ),
      sortable: true,
      sortBy: (e) => e.predicted_probability,
    },
    {
      header: 'Horizon',
      accessor: (e) => <span className="font-mono text-xs">k={e.prediction_horizon}</span>,
    },
    {
      header: 'True Outcome',
      accessor: (e) =>
        e.true_label !== undefined ? (
          e.true_label === 1 ? (
            <span className="text-xs px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40">
              Failure (1)
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
              Normal (0)
            </span>
          )
        ) : (
          <span className="text-slate-600 text-xs">-</span>
        ),
    },
    {
      header: 'Explanation Method',
      accessor: (e) => (
        <span className="font-mono text-xs text-purple-300 bg-purple-950/30 px-2 py-0.5 rounded border border-purple-900/30">
          {e.explanation_method}
        </span>
      ),
    },
    {
      header: 'Top Attributed Feature',
      accessor: (e) => (
        <span className="text-xs font-mono text-cyan-300">
          {e.important_features?.[0]?.feature_name || 'N/A'}
        </span>
      ),
    },
    {
      header: 'Actions',
      accessor: (e) => (
        <Link
          to={`/explainability/${e.explanation_id}`}
          className="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
        >
          Inspect Case
        </Link>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Explainability & Failure Attribution</h1>
          <p className="text-xs text-slate-400 mt-1">
            Interpreting why AgentGuard predicted an increased probability of impending cascade
          </p>
        </div>
        {explanations.length > 0 && (
          <ExportButton
            data={filteredExplanations}
            filename="explainability_cases"
            label="Export Explanations"
          />
        )}
      </div>

      {/* Methodology Disclaimer */}
      <div className="p-3.5 rounded bg-slate-900/90 border border-cyan-500/30 text-xs text-slate-300 flex items-start gap-3">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-cyan-300">Interpretability Framework:</strong> Explanations are derived via integrated
          gradients, temporal attention weight decomposition, and graph perturbation saliency. Scores reflect predictive
          relevance towards the failure probability output, not physical causality.
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Case Explanations"
          value={explanations.length.toString()}
          subtext="Available empirical cases"
          icon={<Brain className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="True Positive Detections"
          value={explanations.filter((e) => e.case_type?.includes('true_positive') || (e.predicted_label === 1 && e.true_label === 1)).length.toString()}
          subtext="Early failure alarms confirmed"
          status="success"
        />
        <MetricCard
          label="False Alarms (FP)"
          value={explanations.filter((e) => e.case_type?.includes('false_positive') || (e.predicted_label === 1 && e.true_label === 0)).length.toString()}
          subtext="Benign runs flagged by model"
          status="warning"
          icon={<ShieldAlert className="w-4 h-4 text-amber-400" />}
        />
        <MetricCard
          label="Explanation Methods"
          value="3"
          subtext="Integrated Gradients, Attention, Perturbation"
          icon={<Sparkles className="w-4 h-4 text-purple-400" />}
        />
      </div>

      {/* Saliency & Attribution Overview Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Features Chart */}
        <Card
          title="Global Top Attributed Features"
          subtitle="Mean feature attribution score across all logged failure explanations"
          icon={<Sparkles className="w-4 h-4 text-cyan-400" />}
        >
          {aggregatedFeatures.length > 0 ? (
            <ChartContainer height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={aggregatedFeatures} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#64748b" fontSize={10} domain={[0, 'dataMax + 0.05']} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={10} width={120} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-slate-900 border border-slate-800 p-2 rounded text-xs font-mono">
                            <div className="text-slate-300 font-bold">{d.name}</div>
                            <div className="text-cyan-400">Mean Importance: {d.score}</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="score" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <EmptyState title="No Features Logged" message="No feature attribution records found." />
          )}
        </Card>

        {/* Top Agent Contributions Chart */}
        <Card
          title="Most Frequently Implicated Agents"
          subtitle="Mean agent structural attribution score across case studies"
          icon={<Users className="w-4 h-4 text-purple-400" />}
        >
          {aggregatedAgents.length > 0 ? (
            <ChartContainer height={260}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={aggregatedAgents} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#64748b" fontSize={10} domain={[0, 'dataMax + 0.05']} />
                  <YAxis type="category" dataKey="agent_id" stroke="#64748b" fontSize={10} width={90} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-slate-900 border border-slate-800 p-2 rounded text-xs font-mono">
                            <div className="text-slate-300 font-bold">{d.agent_id} ({d.role})</div>
                            <div className="text-purple-400">Attribution Score: {d.score}</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="score" fill="#a855f7" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <EmptyState title="No Agents Logged" message="No agent contribution records found." />
          )}
        </Card>
      </div>

      {/* Filter Tabs for Case Studies */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        {[
          { id: 'all', label: 'All Cases' },
          { id: 'true_positive', label: 'True Positives (Detected Failures)' },
          { id: 'false_positive', label: 'False Positives (False Alarms)' },
          { id: 'false_negative', label: 'False Negatives (Missed Failures)' },
          { id: 'early_warning', label: 'Early Warnings (High Lead Time)' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setCaseFilter(tab.id)}
            className={`text-xs px-3 py-1.5 rounded transition font-medium ${
              caseFilter === tab.id
                ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Case Studies Table */}
      <Card
        title="Empirical Case Study Repository"
        subtitle="Individual prediction instances with granular attribution decomposition"
      >
        {loading ? (
          <LoadingState message="Fetching case explanations from API..." />
        ) : error ? (
          <ErrorState title="Error Loading Explanations" message={error} onRetry={fetchExplanations} />
        ) : (
          <DataTable
            columns={columns}
            data={filteredExplanations}
            keyField="explanation_id"
            emptyMessage="No case studies match the selected filter."
            pageSize={10}
            searchPlaceholder="Search by Explanation ID, Run ID, or Feature..."
          />
        )}
      </Card>
    </div>
  );
};

export default ExplainabilityPage;
