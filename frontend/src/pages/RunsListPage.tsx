import React, { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { getRuns } from '../api/runs';
import type { RunResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import DataTable, { type Column } from '../components/common/DataTable';
import FilterBar, { type FilterField } from '../components/common/FilterBar';
import { StatusBadge } from '../components/common/Badges';
import { LoadingState, ErrorState } from '../components/common/States';
import ExportButton from '../components/common/ExportButton';
import { PlaySquare, AlertTriangle, Layers, Users, ExternalLink } from 'lucide-react';

export const RunsListPage: React.FC = () => {
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Record<string, string>>({
    task: '',
    topology: '',
    status: '',
    agent_count: '',
  });

  const filterFields: FilterField[] = [
    {
      key: 'task',
      label: 'Task',
      type: 'select',
      value: filters.task,
      placeholder: 'All Tasks',
      options: [
        { label: 'Collaborative Coding', value: 'collaborative_coding' },
        { label: 'Financial Analysis', value: 'financial_analysis' },
        { label: 'Customer Support', value: 'customer_support' },
        { label: 'Scientific Research', value: 'scientific_research' },
      ],
    },
    {
      key: 'topology',
      label: 'Topology',
      type: 'select',
      value: filters.topology,
      placeholder: 'All Topologies',
      options: [
        { label: 'Fully Connected', value: 'fully_connected' },
        { label: 'Hierarchical', value: 'hierarchical' },
        { label: 'Ring', value: 'ring' },
        { label: 'Star', value: 'star' },
        { label: 'Pipeline', value: 'pipeline' },
      ],
    },
    {
      key: 'status',
      label: 'Status',
      type: 'select',
      value: filters.status,
      placeholder: 'All Statuses',
      options: [
        { label: 'Completed', value: 'completed' },
        { label: 'Failed', value: 'failed' },
        { label: 'Cascaded', value: 'cascaded' },
        { label: 'Running', value: 'running' },
      ],
    },
    {
      key: 'agent_count',
      label: 'Agents',
      type: 'select',
      value: filters.agent_count,
      placeholder: 'Any Count',
      options: [
        { label: '3 Agents', value: '3' },
        { label: '4 Agents', value: '4' },
        { label: '5 Agents', value: '5' },
        { label: '6 Agents', value: '6' },
        { label: '8 Agents', value: '8' },
        { label: '10 Agents', value: '10' },
        { label: '12 Agents', value: '12' },
      ],
    },
  ];

  const handleFilterChange = (id: string, value: string) => {
    setFilters((prev) => ({ ...prev, [id]: value }));
  };

  const handleResetFilters = () => {
    setFilters({ task: '', topology: '', status: '', agent_count: '' });
  };

  const fetchRuns = () => {
    setLoading(true);
    setError(null);

    const apiParams: Record<string, any> = { limit: 100 };
    if (filters.task) apiParams.task = filters.task;
    if (filters.topology) apiParams.topology = filters.topology;
    if (filters.status) apiParams.status = filters.status;
    if (filters.agent_count) apiParams.agent_count = parseInt(filters.agent_count, 10);

    getRuns(apiParams)
      .then((res) => {
        setRuns(res.items || []);
        setTotalCount(res.total ?? (res.items || []).length);
      })
      .catch((err: any) => {
        setError(err.message || 'Failed to fetch simulation runs.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchRuns();
  }, [filters]);

  const summary = useMemo(() => {
    const total = runs.length;
    const failures = runs.filter(
      (r) => r.status === 'failed' || r.has_cascading_failure || (r.failure_count ?? 0) > 0
    ).length;
    const avgAgents = total > 0 ? (runs.reduce((acc, r) => acc + (r.agent_count || 0), 0) / total).toFixed(1) : '0';
    return { total, failures, avgAgents };
  }, [runs]);

  const columns: Column<RunResponse>[] = [
    {
      header: 'Run ID',
      accessor: (r: RunResponse) => {
        const runId = r.id || r.run_id || '';
        return (
          <Link
            to={`/runs/${runId}`}
            className="font-mono text-xs font-semibold text-cyan-400 hover:text-cyan-300 hover:underline flex items-center gap-1.5"
          >
            {runId.slice(0, 16)}...
            <ExternalLink className="w-3 h-3 opacity-60" />
          </Link>
        );
      },
      sortable: true,
      sortBy: (r: RunResponse) => r.id || r.run_id || '',
    },
    {
      header: 'Task Environment',
      accessor: (r: RunResponse) => <span className="text-slate-200 capitalize">{r.task.replace(/_/g, ' ')}</span>,
      sortable: true,
      sortBy: (r: RunResponse) => r.task,
    },
    {
      header: 'Topology',
      accessor: (r: RunResponse) => (
        <span className="font-mono text-xs bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-300">
          {r.topology}
        </span>
      ),
      sortable: true,
      sortBy: (r: RunResponse) => r.topology,
    },
    {
      header: 'Agents',
      accessor: (r: RunResponse) => <span className="font-mono text-xs text-slate-300">{r.agent_count}</span>,
      sortable: true,
      sortBy: (r: RunResponse) => r.agent_count,
    },
    {
      header: 'Seed',
      accessor: (r: RunResponse) => <span className="font-mono text-xs text-slate-400">{r.random_seed ?? r.seed}</span>,
    },
    {
      header: 'Duration / Steps',
      accessor: (r: RunResponse) => (
        <span className="font-mono text-xs text-slate-300">
          {r.total_steps ? `${r.total_steps} steps` : r.duration_seconds ? `${r.duration_seconds.toFixed(1)}s` : '-'}
        </span>
      ),
      sortable: true,
      sortBy: (r: RunResponse) => r.total_steps ?? r.duration_seconds ?? 0,
    },
    {
      header: 'Cascaded',
      accessor: (r: RunResponse) => {
        if (!r.has_cascading_failure) return <span className="text-slate-600 text-xs">No</span>;
        return (
          <span className="inline-flex items-center gap-1 text-xs font-mono font-semibold text-rose-400 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-900/40">
            <AlertTriangle className="w-3 h-3" />
            Yes
          </span>
        );
      },
      sortable: true,
      sortBy: (r: RunResponse) => (r.has_cascading_failure ? 1 : 0),
    },
    {
      header: 'Cascade Step',
      accessor: (r: RunResponse) => {
        const step = r.cascading_failure_step ?? r.cascade_step;
        return step !== undefined && step !== null ? (
          <span className="font-mono text-xs text-rose-400">Step {step}</span>
        ) : (
          <span className="text-slate-600 text-xs">None</span>
        );
      },
    },
    {
      header: 'Status',
      accessor: (r: RunResponse) => <StatusBadge status={r.status} />,
      sortable: true,
      sortBy: (r: RunResponse) => r.status,
    },
    {
      header: 'Actions',
      accessor: (r: RunResponse) => {
        const runId = r.id || r.run_id || '';
        return (
          <Link
            to={`/runs/${runId}`}
            className="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            Inspect Run
          </Link>
        );
      },
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">Simulation Runs</h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical multi-agent execution trajectories, fault injections, and system cascade telemetry
          </p>
        </div>
        <div className="flex items-center gap-2">
          {runs.length > 0 && (
            <ExportButton data={runs} filename="simulation_runs" label="Export Runs Data" />
          )}
        </div>
      </div>

      {/* KPI Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Runs Displayed"
          value={runs.length.toString()}
          subtext={`Backend total: ${totalCount}`}
          icon={<PlaySquare className="w-4 h-4 text-cyan-400" />}
        />
        <MetricCard
          label="Runs with Failures"
          value={summary.failures.toString()}
          subtext={runs.length > 0 ? `${((summary.failures / runs.length) * 100).toFixed(1)}% failure rate` : ''}
          status={summary.failures > 0 ? 'warning' : 'neutral'}
          icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}
        />
        <MetricCard
          label="Avg Agents / Run"
          value={summary.avgAgents}
          subtext="Configured multi-agent team size"
          icon={<Users className="w-4 h-4 text-indigo-400" />}
        />
        <MetricCard
          label="Topologies Covered"
          value="5"
          subtext="fc, hierarchical, ring, star, pipeline"
          icon={<Layers className="w-4 h-4 text-purple-400" />}
        />
      </div>

      {/* Filters Bar */}
      <FilterBar fields={filterFields} onChange={handleFilterChange} onReset={handleResetFilters} />

      {/* Runs Table */}
      <Card
        title="Simulation Trajectories"
        subtitle="Sort and filter individual execution traces"
      >
        {loading ? (
          <LoadingState message="Fetching simulation runs from API..." />
        ) : error ? (
          <ErrorState title="Error Loading Runs" message={error} onRetry={fetchRuns} />
        ) : (
          <DataTable
            columns={columns}
            data={runs}
            keyField="id"
            emptyMessage="No simulation runs match the selected filters."
            pageSize={15}
            searchPlaceholder="Search runs by ID or task..."
          />
        )}
      </Card>
    </div>
  );
};

export default RunsListPage;
