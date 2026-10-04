import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getExplanationById } from '../api/explainability';
import { getRunById, getRunGraph, getRunEvents, getRunFailures, getRunPredictions } from '../api/runs';
import type {
  ExplanationResponse,
  RunDetailResponse,
  RunGraphResponse,
  EventResponse,
  RunFailureResponse,
  PredictionResponse,
} from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import ExportButton from '../components/common/ExportButton';
import GraphViewer from '../components/graphs/GraphViewer';
import FailureTimeline from '../components/timeline/FailureTimeline';
import DataTable, { type Column } from '../components/common/DataTable';
import {
  ArrowLeft,
  Brain,
  Sparkles,
  Users,
  Network,
  Info,
  Clock,
} from 'lucide-react';

export const CaseStudyDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [run, setRun] = useState<RunDetailResponse | null>(null);
  const [graphData, setGraphData] = useState<RunGraphResponse | null>(null);
  const [events, setEvents] = useState<EventResponse[]>([]);
  const [failures, setFailures] = useState<RunFailureResponse[]>([]);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'attributions' | 'graph' | 'timeline'>('attributions');

  useEffect(() => {
    if (!id) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    getExplanationById(id)
      .then((exp) => {
        if (!isMounted) return;
        setExplanation(exp);

        // Fetch associated trajectory and graph
        return Promise.all([
          getRunById(exp.run_id).catch(() => null),
          getRunGraph(exp.run_id).catch(() => null),
          getRunEvents(exp.run_id, { limit: 100 }).catch(() => ({ items: [], total: 0 })),
          getRunFailures(exp.run_id, { limit: 50 }).catch(() => ({ items: [], total: 0 })),
          getRunPredictions(exp.run_id, { limit: 100 }).catch(() => ({ items: [], total: 0 })),
        ]).then(([runRes, graphRes, eventsRes, failuresRes, predsRes]) => {
          if (!isMounted) return;
          setRun(runRes);
          setGraphData(graphRes);
          setEvents(eventsRes.items || []);
          setFailures(failuresRes.items || []);
          setPredictions(predsRes.items || []);
        });
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setError(err.message || 'Failed to load case explanation.');
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [id]);

  if (loading) {
    return <LoadingState message="Decomposing model attribution, graph topology, and risk timeline..." />;
  }

  if (error || !explanation) {
    return (
      <ErrorState
        title="Case Study Not Found"
        message={error || `Explanation '${id}' could not be located.`}
        onRetry={() => window.location.reload()}
      />
    );
  }

  const featureColumns: Column<any>[] = [
    {
      header: 'Feature Name',
      accessor: (f) => <span className="font-mono text-xs text-slate-200 font-semibold">{f.feature_name}</span>,
      sortable: true,
      sortBy: (f) => f.feature_name,
    },
    {
      header: 'Attribution Score',
      accessor: (f) => (
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-400"
              style={{ width: `${Math.min(100, Math.round(f.importance_score * 100))}%` }}
            />
          </div>
          <span className="font-mono text-xs text-cyan-300 font-bold">{f.importance_score.toFixed(4)}</span>
        </div>
      ),
      sortable: true,
      sortBy: (f) => f.importance_score,
    },
    {
      header: 'Directional Influence',
      accessor: (f) => (
        <span
          className={`text-xs px-2 py-0.5 rounded font-mono ${
            f.direction === 'increases_risk' || f.importance_score > 0
              ? 'bg-rose-950/40 text-rose-300 border border-rose-900/40'
              : 'bg-emerald-950/40 text-emerald-300 border border-emerald-900/40'
          }`}
        >
          {f.direction || (f.importance_score > 0 ? '+ Risk' : '- Risk')}
        </span>
      ),
    },
  ];

  const agentColumns: Column<any>[] = [
    {
      header: 'Agent ID',
      accessor: (a) => <span className="font-mono text-xs text-slate-200 font-semibold">{a.agent_id}</span>,
      sortable: true,
      sortBy: (a) => a.agent_id,
    },
    {
      header: 'Role',
      accessor: (a) => <span className="text-xs text-slate-400 capitalize">{a.role || 'Worker'}</span>,
    },
    {
      header: 'Failure Role',
      accessor: (a) => (
        <span className="text-xs font-mono text-amber-400">{a.failure_role || 'Propagator / Node'}</span>
      ),
    },
    {
      header: 'Attribution Score',
      accessor: (a) => (
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-400"
              style={{ width: `${Math.min(100, Math.round(a.importance_score * 100))}%` }}
            />
          </div>
          <span className="font-mono text-xs text-purple-300 font-bold">{a.importance_score.toFixed(4)}</span>
        </div>
      ),
      sortable: true,
      sortBy: (a) => a.importance_score,
    },
  ];

  const edgeColumns: Column<any>[] = [
    {
      header: 'Interaction Edge',
      accessor: (e) => (
        <span className="font-mono text-xs text-cyan-300 font-semibold">
          {e.source_agent} <span className="text-slate-500">→</span> {e.target_agent}
        </span>
      ),
    },
    {
      header: 'Edge Attribution',
      accessor: (e) => (
        <span className="font-mono text-xs text-indigo-300 font-bold">{e.importance_score.toFixed(4)}</span>
      ),
      sortable: true,
      sortBy: (e) => e.importance_score,
    },
    {
      header: 'Interaction Count',
      accessor: (e) => <span className="font-mono text-xs text-slate-300">{e.interaction_count ?? 1}</span>,
    },
    {
      header: 'Contradiction Rate',
      accessor: (e) => (
        <span className="font-mono text-xs text-rose-400">
          {e.contradiction_rate !== undefined ? `${(e.contradiction_rate * 100).toFixed(0)}%` : '0%'}
        </span>
      ),
    },
  ];

  const eventColumns: Column<any>[] = [
    {
      header: 'Step',
      accessor: (ev) => <span className="font-mono text-xs text-slate-300">Step {ev.step_idx}</span>,
      sortable: true,
      sortBy: (ev) => ev.step_idx,
    },
    {
      header: 'Timestamp',
      accessor: (ev) => <span className="font-mono text-xs text-slate-400">{ev.timestamp.toFixed(2)}s</span>,
    },
    {
      header: 'Interaction',
      accessor: (ev) => (
        <span className="font-mono text-xs text-slate-300">
          {ev.source_agent} → {ev.target_agent}
        </span>
      ),
    },
    {
      header: 'Saliency Score',
      accessor: (ev) => (
        <span className="font-mono text-xs text-amber-400 font-semibold">{ev.importance_score.toFixed(4)}</span>
      ),
      sortable: true,
      sortBy: (ev) => ev.importance_score,
    },
    {
      header: 'Event Summary',
      accessor: (ev) => <span className="text-xs text-slate-300">{ev.summary || '-'}</span>,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            to="/explainability"
            className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100 tracking-tight">Case Study: {explanation.explanation_id}</h1>
              <span className="text-xs px-2 py-0.5 rounded border uppercase font-mono font-semibold bg-cyan-950/40 text-cyan-300 border-cyan-800/40">
                {explanation.case_type?.replace(/_/g, ' ') || 'Case Study'}
              </span>
            </div>
            <p className="text-xs font-mono text-slate-500 mt-0.5">
              Run:{' '}
              <Link to={`/runs/${explanation.run_id}`} className="text-cyan-400 hover:underline">
                {explanation.run_id}
              </Link>{' '}
              · Method: <span className="text-slate-300">{explanation.explanation_method}</span> · Model:{' '}
              <span className="text-slate-300">{explanation.model_name}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ExportButton data={explanation} filename={`explanation_${explanation.explanation_id}`} label="Export Case JSON" />
        </div>
      </div>

      {/* Case Metrics Banner */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Impending Risk P(Y=1)"
          value={explanation.predicted_probability.toFixed(4)}
          subtext={`Threshold τ = ${explanation.threshold.toFixed(2)}`}
          status={explanation.predicted_probability >= explanation.threshold ? 'danger' : 'success'}
        />
        <MetricCard
          label="Prediction Horizon"
          value={`k = ${explanation.prediction_horizon} steps`}
          subtext="Lookahead window"
          icon={<Clock className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="Model Decision"
          value={explanation.predicted_label === 1 ? 'Warning (1)' : 'Normal (0)'}
          status={explanation.predicted_label === 1 ? 'warning' : 'neutral'}
        />
        <MetricCard
          label="Actual Outcome"
          value={
            explanation.true_label !== undefined
              ? explanation.true_label === 1
                ? 'Cascade Failure (1)'
                : 'Clean Execution (0)'
              : 'Pending'
          }
          status={explanation.true_label === 1 ? 'danger' : 'success'}
        />
        <MetricCard
          label="Detection Status"
          value={
            explanation.predicted_label === explanation.true_label
              ? explanation.predicted_label === 1
                ? 'True Positive'
                : 'True Negative'
              : explanation.predicted_label === 1
              ? 'False Positive'
              : 'False Negative'
          }
          status={explanation.predicted_label === explanation.true_label ? 'success' : 'danger'}
        />
      </div>

      {/* Scientific High-Level Summary */}
      {explanation.high_level_summary && (
        <div className="p-4 rounded bg-slate-900 border border-slate-800 text-xs text-slate-300 space-y-1">
          <div className="text-slate-400 font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Synthesized Saliency Summary
          </div>
          <p className="leading-relaxed text-slate-200">{explanation.high_level_summary}</p>
        </div>
      )}

      {/* Non-Causality Disclaimer */}
      <div className="p-3.5 rounded bg-slate-950 border border-amber-500/20 text-xs text-slate-400 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-amber-300">Non-Causality Disclaimer:</strong>{' '}
          {explanation.causality_disclaimer ||
            'Attribution scores represent predictive gradient and attention saliency toward the failure classification output. They describe model reliance and empirical correlation, not mechanistic or physical causation.'}
        </div>
      </div>

      {/* View Switcher Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('attributions')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'attributions'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Brain className="w-3.5 h-3.5" />
          Feature & Agent Attributions
        </button>
        <button
          onClick={() => setActiveTab('graph')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'graph'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Network className="w-3.5 h-3.5" />
          Case Interaction Graph
        </button>
        <button
          onClick={() => setActiveTab('timeline')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'timeline'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Clock className="w-3.5 h-3.5" />
          Trajectory Timeline & Cascade
        </button>
      </div>

      {/* Tab 1: Attributions Breakdown */}
      {activeTab === 'attributions' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Features */}
            <Card
              title="Ranked Predictive Features"
              subtitle="Feature importance weights computed via integrated gradients"
              icon={<Sparkles className="w-4 h-4 text-cyan-400" />}
            >
              <DataTable
                columns={featureColumns}
                data={explanation.important_features || []}
                keyField="feature_name"
                emptyMessage="No feature attributions recorded for this case."
                pageSize={6}
              />
            </Card>

            {/* Implicated Agents */}
            <Card
              title="Implicated Agent Contributions"
              subtitle="Agent node saliency within the dynamic interaction topology"
              icon={<Users className="w-4 h-4 text-purple-400" />}
            >
              <DataTable
                columns={agentColumns}
                data={explanation.important_agents || []}
                keyField="agent_id"
                emptyMessage="No agent contributions recorded for this case."
                pageSize={6}
              />
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Implicated Edges */}
            <Card
              title="Implicated Interaction Channels (Edges)"
              subtitle="Pairwise communication channels contributing to failure risk"
              icon={<Network className="w-4 h-4 text-indigo-400" />}
            >
              <DataTable
                columns={edgeColumns}
                data={explanation.important_edges || []}
                keyField="source_agent"
                emptyMessage="No edge attributions recorded for this case."
                pageSize={6}
              />
            </Card>

            {/* Salient Historical Events */}
            <Card
              title="Salient Historical Telemetry Events"
              subtitle="Chronological messages in the trajectory identified as pivotal"
              icon={<Clock className="w-4 h-4 text-amber-400" />}
            >
              <DataTable
                columns={eventColumns}
                data={explanation.important_events || []}
                keyField="step_idx"
                emptyMessage="No specific event attributions recorded for this case."
                pageSize={6}
              />
            </Card>
          </div>
        </div>
      )}

      {/* Tab 2: Graph Viewer */}
      {activeTab === 'graph' && (
        <Card
          title="Dynamic Interaction Graph (Trajectory Context)"
          subtitle={`Multi-agent topology for Run ${explanation.run_id}. Compare structural connections with attributed agents.`}
        >
          {graphData && graphData.nodes && graphData.nodes.length > 0 ? (
            <GraphViewer graph={graphData} maxSteps={run?.total_steps || 30} />
          ) : (
            <EmptyState
              title="Interaction Graph Unavailable"
              message="No temporal graph data found for this case's simulation run."
            />
          )}
        </Card>
      )}

      {/* Tab 3: Timeline */}
      {activeTab === 'timeline' && (
        <Card
          title="Trajectory Timeline & Warning Progression"
          subtitle="Time step alignment showing when the model raised the alarm relative to faults and cascade onset"
        >
          <FailureTimeline
            events={events}
            failures={failures}
            predictions={predictions}
            cascadeStep={run?.cascade_step}
            totalSteps={run?.total_steps || 30}
          />
        </Card>
      )}
    </div>
  );
};

export default CaseStudyDetailPage;
