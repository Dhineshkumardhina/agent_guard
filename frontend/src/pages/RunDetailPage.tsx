import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  getRunById,
  getRunEvents,
  getRunFailures,
  getRunGraph,
  getRunPredictions,
} from '../api/runs';
import type {
  RunDetailResponse,
  EventResponse,
  RunFailureResponse,
  RunGraphResponse,
  PredictionResponse,
} from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { StatusBadge } from '../components/common/Badges';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import GraphViewer from '../components/graphs/GraphViewer';
import FailureTimeline from '../components/timeline/FailureTimeline';
import {
  ArrowLeft,
  Share2,
  Clock,
  Layers,
  Users,
  AlertTriangle,
  Activity,
  ListFilter,
} from 'lucide-react';

export const RunDetailPage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();

  const [run, setRun] = useState<RunDetailResponse | null>(null);
  const [events, setEvents] = useState<EventResponse[]>([]);
  const [failures, setFailures] = useState<RunFailureResponse[]>([]);
  const [graphData, setGraphData] = useState<RunGraphResponse | null>(null);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'graph' | 'timeline' | 'events' | 'failures'>('graph');

  useEffect(() => {
    if (!runId) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    Promise.all([
      getRunById(runId),
      getRunEvents(runId, { limit: 200 }).catch(() => ({ items: [], total: 0 })),
      getRunFailures(runId, { limit: 100 }).catch(() => ({ items: [], total: 0 })),
      getRunGraph(runId).catch(() => null),
      getRunPredictions(runId, { limit: 100 }).catch(() => ({ items: [], total: 0 })),
    ])
      .then(([runRes, eventsRes, failuresRes, graphRes, predRes]) => {
        if (!isMounted) return;
        setRun(runRes);
        setEvents(eventsRes.items || []);
        setFailures(failuresRes.items || []);
        setGraphData(graphRes);
        setPredictions(predRes.items || []);
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setError(err.message || 'Failed to load simulation run details.');
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [runId]);

  if (loading) {
    return <LoadingState message={`Fetching trajectory and interaction graph for run ${runId}...`} />;
  }

  if (error || !run) {
    return (
      <ErrorState
        title="Simulation Run Not Found"
        message={error || `Run '${runId}' could not be loaded.`}
        onRetry={() => window.location.reload()}
      />
    );
  }

  const effectiveRunId = run.id || run.run_id || runId || '';
  const effectiveSeed = run.random_seed ?? run.seed ?? 42;
  const effectiveTotalSteps = run.total_steps ?? (run.duration_seconds ? Math.round(run.duration_seconds) : 30);
  const effectiveCascadeStep = run.cascading_failure_step ?? run.cascade_step;

  const eventColumns: Column<EventResponse>[] = [
    {
      header: 'Step',
      accessor: (e) => <span className="font-mono text-xs text-slate-300">{e.step_idx}</span>,
      sortable: true,
      sortBy: (e) => e.step_idx,
    },
    {
      header: 'Timestamp',
      accessor: (e) => <span className="font-mono text-xs text-slate-400">{e.timestamp.toFixed(2)}s</span>,
      sortable: true,
      sortBy: (e) => e.timestamp,
    },
    {
      header: 'Sender',
      accessor: (e) => <span className="font-mono text-xs text-cyan-300 font-semibold">{e.source_agent || e.sender}</span>,
    },
    {
      header: 'Receiver',
      accessor: (e) => <span className="font-mono text-xs text-purple-300 font-semibold">{e.target_agent || e.receiver}</span>,
    },
    {
      header: 'Type',
      accessor: (e) => (
        <span className="font-mono text-xs bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-300">
          {e.event_type}
        </span>
      ),
    },
    {
      header: 'Contradiction',
      accessor: (e) =>
        e.is_contradiction || e.contradiction_score > 0.5 ? (
          <span className="text-xs px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40">
            Yes
          </span>
        ) : (
          <span className="text-slate-600 text-xs">No</span>
        ),
    },
    {
      header: 'Latency',
      accessor: (e) => (
        <span className="font-mono text-xs text-slate-400">
          {e.latency_ms ? `${e.latency_ms.toFixed(1)}ms` : e.latency ? `${(e.latency * 1000).toFixed(0)}ms` : '-'}
        </span>
      ),
    },
    {
      header: 'Summary / Payload',
      accessor: (e) => (
        <div className="max-w-xs truncate text-xs text-slate-300 font-mono" title={e.summary || JSON.stringify(e.metadata_json || {})}>
          {e.summary || JSON.stringify(e.metadata_json || {})}
        </div>
      ),
    },
  ];

  const failureColumns: Column<RunFailureResponse>[] = [
    {
      header: 'Step',
      accessor: (f) => <span className="font-mono text-xs text-rose-400 font-semibold">Step {f.step_idx}</span>,
      sortable: true,
      sortBy: (f) => f.step_idx,
    },
    {
      header: 'Agent',
      accessor: (f) => <span className="font-mono text-xs text-slate-200 font-semibold">{f.originating_agent || f.agent_id}</span>,
    },
    {
      header: 'Failure Type',
      accessor: (f) => (
        <span className="font-mono text-xs bg-rose-950/40 border border-rose-900/40 px-2 py-0.5 rounded text-rose-300">
          {f.failure_type}
        </span>
      ),
    },
    {
      header: 'Severity Level',
      accessor: (f) => (
        <span className="font-mono text-xs text-amber-400">
          Level {f.failure_level ?? f.severity ?? 1}
        </span>
      ),
    },
    {
      header: 'Cascaded',
      accessor: (f) =>
        f.cascaded || run.has_cascading_failure ? (
          <span className="text-xs font-semibold text-rose-400">Yes (System Failure)</span>
        ) : (
          <span className="text-xs text-slate-500">Contained</span>
        ),
    },
    {
      header: 'Description',
      accessor: (f) => <span className="text-xs text-slate-300">{f.description || '-'}</span>,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Action */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            to="/runs"
            className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100 tracking-tight">Run: {effectiveRunId}</h1>
              <StatusBadge status={run.status} />
            </div>
            <p className="text-xs font-mono text-slate-500 mt-0.5">
              Task: <span className="text-slate-300">{run.task}</span> · Topology:{' '}
              <span className="text-slate-300">{run.topology}</span> · Seed: {effectiveSeed}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {graphData && (
            <ExportButton data={graphData} filename={`graph_${effectiveRunId}`} label="Export Graph" />
          )}
          {events.length > 0 && (
            <ExportButton data={events} filename={`events_${effectiveRunId}`} label="Export Telemetry" />
          )}
        </div>
      </div>

      {/* Trajectory KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Total Agents"
          value={run.agent_count.toString()}
          subtext="Configured node count"
          icon={<Users className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="Topology"
          value={run.topology}
          icon={<Layers className="w-4 h-4 text-purple-400" />}
        />
        <MetricCard
          label="Total Steps"
          value={effectiveTotalSteps.toString()}
          subtext={run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s elapsed` : ''}
          icon={<Clock className="w-4 h-4 text-emerald-400" />}
        />
        <MetricCard
          label="Telemetry Events"
          value={events.length.toString()}
          subtext="Interaction messages"
          icon={<Activity className="w-4 h-4 text-indigo-400" />}
        />
        <MetricCard
          label="Failures Logged"
          value={failures.length.toString()}
          subtext={effectiveCascadeStep !== undefined && effectiveCascadeStep !== null ? `Cascade at step ${effectiveCascadeStep}` : 'No cascade'}
          status={failures.length > 0 ? 'danger' : 'success'}
          icon={<AlertTriangle className="w-4 h-4 text-rose-400" />}
        />
      </div>

      {/* Navigation Tabs for Views */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('graph')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'graph'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Share2 className="w-3.5 h-3.5" />
          Temporal Interaction Graph
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
          Failure & Cascade Timeline
        </button>
        <button
          onClick={() => setActiveTab('events')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'events'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <ListFilter className="w-3.5 h-3.5" />
          Telemetry Events ({events.length})
        </button>
        <button
          onClick={() => setActiveTab('failures')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition ${
            activeTab === 'failures'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <AlertTriangle className="w-3.5 h-3.5" />
          Failures & Cascade Details ({failures.length})
        </button>
      </div>

      {/* Tab 1: Graph Visualization */}
      {activeTab === 'graph' && (
        <Card
          title="Temporal Interaction Graph Viewer"
          subtitle="Multi-agent topology state at chosen time step. Click nodes and edges to inspect dynamic attributes."
        >
          {graphData && graphData.nodes && graphData.nodes.length > 0 ? (
            <GraphViewer graph={graphData} maxSteps={effectiveTotalSteps} />
          ) : (
            <EmptyState
              title="Interaction Graph Unavailable"
              message="No graph representation could be extracted for this simulation run."
            />
          )}
        </Card>
      )}

      {/* Tab 2: Timeline */}
      {activeTab === 'timeline' && (
        <Card
          title="Early Warning & Cascade Progression Timeline"
          subtitle="Chronological sequence of normal interactions, injected faults, model warning thresholds, and actual failure onset"
        >
          <FailureTimeline
            events={events}
            failures={failures}
            predictions={predictions}
            cascadeStep={effectiveCascadeStep}
            totalSteps={effectiveTotalSteps}
          />
        </Card>
      )}

      {/* Tab 3: Telemetry Events Table */}
      {activeTab === 'events' && (
        <Card
          title="Raw Telemetry Message Stream"
          subtitle="Inter-agent communications, task instructions, verification checks, and state transitions"
          action={
            events.length > 0 ? (
              <ExportButton data={events} filename={`telemetry_${effectiveRunId}`} label="Export Events" />
            ) : undefined
          }
        >
          <DataTable
            columns={eventColumns}
            data={events}
            keyField="id"
            emptyMessage="No telemetry messages recorded in this trajectory."
            pageSize={15}
            searchPlaceholder="Search events by sender or receiver..."
          />
        </Card>
      )}

      {/* Tab 4: Failures Table */}
      {activeTab === 'failures' && (
        <Card
          title="Failure Injections & Cascade Propagation Log"
          subtitle="Ground-truth failure events injected and systemic cascades triggered during execution"
          action={
            failures.length > 0 ? (
              <ExportButton data={failures} filename={`failures_${effectiveRunId}`} label="Export Failures" />
            ) : undefined
          }
        >
          <DataTable
            columns={failureColumns}
            data={failures}
            keyField="id"
            emptyMessage="No failure events recorded in this trajectory (clean successful run)."
            pageSize={10}
            searchPlaceholder="Search failures by agent ID or type..."
          />
        </Card>
      )}
    </div>
  );
};

export default RunDetailPage;
