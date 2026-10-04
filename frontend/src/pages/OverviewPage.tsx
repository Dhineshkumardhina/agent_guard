import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Layers,
  BarChart3,
  ArrowRight,
  Database,
  Cpu,
} from "lucide-react";
import { Card } from "../components/common/Card";
import { MetricCard } from "../components/common/MetricCard";
import { LoadingState, ErrorState } from "../components/common/States";
import { StatusBadge } from "../components/common/Badges";
import {
  getRuns,
  getExperiments,
  getModelComparison,
} from "../api";
import type { RunResponse, ExperimentResponse, ModelComparisonItem } from "../types";

export const OverviewPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [runsCount, setRunsCount] = useState<number>(0);
  const [failedRunsCount, setFailedRunsCount] = useState<number>(0);
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([]);
  const [comparisonModels, setComparisonModels] = useState<ModelComparisonItem[]>([]);
  const [recentRuns, setRecentRuns] = useState<RunResponse[]>([]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [runsRes, failedRes, expsRes, compRes] = await Promise.all([
        getRuns({ limit: 10 }),
        getRuns({ status: "failed", limit: 1 }),
        getExperiments({ limit: 10 }),
        getModelComparison(1).catch(() => ({ models: [] })),
      ]);

      setRunsCount(runsRes.total);
      setRecentRuns(runsRes.items);
      setFailedRunsCount(failedRes.total);
      setExperiments(expsRes.items);
      setComparisonModels(compRes.models || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load overview data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return <LoadingState message="Initializing research platform overview..." />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={loadData} />;
  }

  // Model families overview definition
  const modelFamilies = [
    {
      family: "Rule-Based",
      description: "Deterministic thresholding on token latency, retries, and errors",
      baseline: comparisonModels.find((m) => m.model_family.toLowerCase().includes("rule")),
      tag: "Non-ML Baseline",
    },
    {
      family: "Classical ML",
      description: "Logistic Regression, Random Forest, XGBoost without graph structure",
      baseline: comparisonModels.find((m) => m.model_family.toLowerCase().includes("classical")),
      tag: "Tabular Feature Baselines",
    },
    {
      family: "Sequence Models",
      description: "Recurrent LSTM & GRU capturing temporal token histories",
      baseline: comparisonModels.find((m) => m.model_family.toLowerCase().includes("sequence") || m.model_name.includes("LSTM")),
      tag: "Temporal Agent Baselines",
    },
    {
      family: "Static GNN",
      description: "GCN & GAT architectures modeling static communication topology",
      baseline: comparisonModels.find((m) => m.model_family.toLowerCase().includes("static") || m.model_name.includes("GAT")),
      tag: "Spatial Graph Baselines",
    },
    {
      family: "Temporal GNN",
      description: "Continuous-time memory-augmented temporal graph neural network (TGN)",
      baseline: comparisonModels.find((m) => m.model_family.toLowerCase().includes("temporal") || m.model_name.includes("Temporal GNN")),
      tag: "Proposed Research Model",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#242b3d] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#f0f3fa]">
            AgentGuard Research Platform
          </h1>
          <p className="mt-1 text-xs text-[#9aa4bf]">
            Temporal Graph-Based Early Warning & Detection of Cascading Failures in Multi-Agent AI Systems
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/experiments"
            className="inline-flex items-center gap-1.5 rounded border border-[#3b82f6] bg-[#3b82f6]/10 px-3 py-1.5 text-xs font-medium text-[#60a5fa] hover:bg-[#3b82f6]/20 transition-colors"
          >
            <span>Model Comparison</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>

      {/* Primary KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-5">
        <MetricCard
          label="Total Experiments"
          value={experiments.length}
          subtext="Benchmark Suites"
          icon={<Layers className="h-4 w-4" />}
        />
        <MetricCard
          label="Models Evaluated"
          value={comparisonModels.length || 9}
          subtext="Across 5 Architectural Families"
          icon={<Cpu className="h-4 w-4" />}
        />
        <MetricCard
          label="Simulation Trajectories"
          value={runsCount}
          subtext={`${failedRunsCount} Cascading Failures`}
          icon={<Activity className="h-4 w-4" />}
        />
        <MetricCard
          label="Prediction Horizons"
          value="K = 1, 3, 5, 10"
          subtext="Discrete Steps Ahead"
          icon={<BarChart3 className="h-4 w-4" />}
        />
        <MetricCard
          label="Dataset Version"
          value="v1.0"
          subtext="generalization_v1"
          icon={<Database className="h-4 w-4" />}
        />
      </div>

      {/* Model Family Overview */}
      <Card
        title="Model Family Performance Overview (Horizon K = 1)"
        subtitle="Empirical performance metrics across the 5 canonical model families"
        action={
          <Link
            to="/experiments"
            className="text-xs text-[#3b82f6] hover:underline flex items-center gap-1"
          >
            <span>Full Comparison</span>
            <ArrowRight className="h-3 w-3" />
          </Link>
        }
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {modelFamilies.map((fam) => (
            <div
              key={fam.family}
              className="flex flex-col justify-between rounded border border-[#242b3d] bg-[#0c0e14] p-3 text-xs"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[#f0f3fa]">{fam.family}</span>
                  <span className="font-mono text-[10px] text-[#5e6984]">{fam.tag}</span>
                </div>
                <p className="mt-1 text-[11px] text-[#9aa4bf]">{fam.description}</p>
              </div>

              <div className="mt-3 border-t border-[#1e2333] pt-2">
                <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                  <div>
                    <span className="text-[#5e6984] block text-[10px]">F1</span>
                    <span className="font-semibold text-[#f0f3fa]">
                      {fam.baseline ? fam.baseline.f1.toFixed(3) : "—"}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#5e6984] block text-[10px]">AUROC</span>
                    <span className="font-semibold text-[#f0f3fa]">
                      {fam.baseline ? fam.baseline.auroc.toFixed(3) : "—"}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#5e6984] block text-[10px]">Lead Time</span>
                    <span className="font-semibold text-[#34d399]">
                      {fam.baseline ? `${fam.baseline.mean_lead_time.toFixed(1)}s` : "—"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Quick Navigation Cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Link
          to="/early-warning"
          className="group block rounded-lg border border-[#242b3d] bg-[#141721] p-4 transition-all hover:border-[#3b82f6]"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[#f0f3fa]">Early Warning Timeline</span>
            <ArrowRight className="h-4 w-4 text-[#5e6984] transition-transform group-hover:translate-x-1 group-hover:text-[#3b82f6]" />
          </div>
          <p className="mt-1 text-xs text-[#9aa4bf]">
            Inspect risk trajectory curves against ground truth cascade onsets and warning lead times.
          </p>
        </Link>

        <Link
          to="/ablations"
          className="group block rounded-lg border border-[#242b3d] bg-[#141721] p-4 transition-all hover:border-[#3b82f6]"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[#f0f3fa]">Ablation Studies</span>
            <ArrowRight className="h-4 w-4 text-[#5e6984] transition-transform group-hover:translate-x-1 group-hover:text-[#3b82f6]" />
          </div>
          <p className="mt-1 text-xs text-[#9aa4bf]">
            Review 120 component removal experiments quantifying spatial, temporal, and semantic contributions.
          </p>
        </Link>

        <Link
          to="/generalization"
          className="group block rounded-lg border border-[#242b3d] bg-[#141721] p-4 transition-all hover:border-[#3b82f6]"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[#f0f3fa]">Generalization Matrix</span>
            <ArrowRight className="h-4 w-4 text-[#5e6984] transition-transform group-hover:translate-x-1 group-hover:text-[#3b82f6]" />
          </div>
          <p className="mt-1 text-xs text-[#9aa4bf]">
            Evaluate model transfer across agent counts (3 to 12), topologies, tasks, and compound shifts.
          </p>
        </Link>
      </div>

      {/* Recent Simulation Trajectories Table */}
      <Card
        title="Recent Simulation Trajectories"
        subtitle="Chronological audit of multi-agent runs and manifested cascade outcomes"
        action={
          <Link to="/runs" className="text-xs text-[#3b82f6] hover:underline">
            View All Runs ({runsCount}) &rarr;
          </Link>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-[#242b3d] text-[11px] font-semibold uppercase text-[#9aa4bf]">
              <tr>
                <th className="py-2">Run ID</th>
                <th className="py-2">Task</th>
                <th className="py-2">Topology</th>
                <th className="py-2">Agents</th>
                <th className="py-2">Outcome</th>
                <th className="py-2">Cascade Step</th>
                <th className="py-2 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2333]">
              {recentRuns.map((r) => (
                <tr key={r.id} className="hover:bg-[#1b2030]/50">
                  <td className="py-2.5 font-mono text-[#f0f3fa]">{r.id}</td>
                  <td className="py-2.5 capitalize text-[#9aa4bf]">{r.task}</td>
                  <td className="py-2.5 capitalize text-[#9aa4bf]">{r.topology}</td>
                  <td className="py-2.5 font-mono text-[#9aa4bf]">{r.agent_count}</td>
                  <td className="py-2.5">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="py-2.5 font-mono text-[#9aa4bf]">
                    {r.cascading_failure_step !== null && r.cascading_failure_step !== undefined
                      ? `Step ${r.cascading_failure_step}`
                      : "—"}
                  </td>
                  <td className="py-2.5 text-right">
                    <Link
                      to={`/runs/${r.id}`}
                      className="text-[#3b82f6] hover:underline font-medium"
                    >
                      Inspect &rarr;
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default OverviewPage;
