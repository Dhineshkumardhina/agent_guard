import React, { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { ChartContainer } from "../components/common/ChartContainer";
import { DataTable, type Column } from "../components/common/DataTable";
import { FilterBar } from "../components/common/FilterBar";
import { ExportButton } from "../components/common/ExportButton";
import { LoadingState, ErrorState } from "../components/common/States";
import { ModelBadge } from "../components/common/Badges";
import { getEvaluations, getModelComparison } from "../api";
import type { EvaluationResponse, ModelComparisonItem } from "../types";

export const ModelComparisonPage: React.FC = () => {
  const [horizon, setHorizon] = useState<number>(1);
  const [modelFilter, setModelFilter] = useState<string>("");

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [allEvaluations, setAllEvaluations] = useState<EvaluationResponse[]>([]);
  const [comparisonModels, setComparisonModels] = useState<ModelComparisonItem[]>([]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [evalsRes, compRes] = await Promise.all([
        getEvaluations({ limit: 100 }),
        getModelComparison(horizon),
      ]);
      setAllEvaluations(evalsRes.items);
      setComparisonModels(compRes.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model comparison");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [horizon]);

  // Prepare chart data: Metrics vs Horizon across canonical models
  const chartDataF1ByHorizon = useMemo(() => {
    const horizons = [1, 3, 5, 10];
    const models = ["Rule-Based", "Logistic Regression", "Random Forest", "LSTM", "GAT", "Temporal GNN"];

    return horizons.map((h) => {
      const entry: Record<string, unknown> = { horizon: `K = ${h}` };
      models.forEach((m) => {
        const found = allEvaluations.find(
          (e) => e.horizon === h && (e.model.toLowerCase() === m.toLowerCase() || e.model.includes(m))
        );
        entry[m] = found ? Number(found.f1.toFixed(3)) : null;
      });
      return entry;
    });
  }, [allEvaluations]);

  const chartDataAuprcByHorizon = useMemo(() => {
    const horizons = [1, 3, 5, 10];
    const models = ["Rule-Based", "Logistic Regression", "Random Forest", "LSTM", "GAT", "Temporal GNN"];

    return horizons.map((h) => {
      const entry: Record<string, unknown> = { horizon: `K = ${h}` };
      models.forEach((m) => {
        const found = allEvaluations.find(
          (e) => e.horizon === h && (e.model.toLowerCase() === m.toLowerCase() || e.model.includes(m))
        );
        entry[m] = found ? Number(found.auprc.toFixed(3)) : null;
      });
      return entry;
    });
  }, [allEvaluations]);

  const chartDataLeadTimeByHorizon = useMemo(() => {
    const horizons = [1, 3, 5, 10];
    const models = ["Rule-Based", "Logistic Regression", "Random Forest", "LSTM", "GAT", "Temporal GNN"];

    return horizons.map((h) => {
      const entry: Record<string, unknown> = { horizon: `K = ${h}` };
      models.forEach((m) => {
        const found = allEvaluations.find(
          (e) => e.horizon === h && (e.model.toLowerCase() === m.toLowerCase() || e.model.includes(m))
        );
        entry[m] = found ? Number(found.mean_lead_time.toFixed(1)) : 0;
      });
      return entry;
    });
  }, [allEvaluations]);

  // Filter comparison items based on user filter selections
  const filteredModels = useMemo(() => {
    return comparisonModels.filter((m) => {
      if (modelFilter && !m.model_name.toLowerCase().includes(modelFilter.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [comparisonModels, modelFilter]);

  // Columns definition for DataTable
  const columns: Column<ModelComparisonItem>[] = [
    {
      key: "model_name",
      header: "Model Architecture",
      render: (item) => (
        <div className="flex items-center gap-2">
          <ModelBadge model={item.model_name} />
          <span className="font-mono text-[11px] text-[#5e6984]">{item.model_family}</span>
        </div>
      ),
    },
    {
      key: "precision",
      header: "Precision",
      align: "right",
      render: (item) => <span className="font-mono">{item.precision.toFixed(4)}</span>,
    },
    {
      key: "recall",
      header: "Recall",
      align: "right",
      render: (item) => <span className="font-mono">{item.recall.toFixed(4)}</span>,
    },
    {
      key: "f1",
      header: "F1 Score",
      align: "right",
      render: (item) => (
        <span className="font-mono font-semibold text-[#f0f3fa]">{item.f1.toFixed(4)}</span>
      ),
    },
    {
      key: "auroc",
      header: "AUROC",
      align: "right",
      render: (item) => <span className="font-mono">{item.auroc.toFixed(4)}</span>,
    },
    {
      key: "auprc",
      header: "AUPRC",
      align: "right",
      render: (item) => <span className="font-mono">{item.auprc.toFixed(4)}</span>,
    },
    {
      key: "false_alarm_rate",
      header: "False Alarm Rate",
      align: "right",
      render: (item) => (
        <span className="font-mono text-[#9aa4bf]">{item.false_alarm_rate.toFixed(4)}</span>
      ),
    },
    {
      key: "mean_lead_time",
      header: "Mean Lead Time",
      align: "right",
      render: (item) => (
        <span className="font-mono text-[#34d399] font-medium">
          {item.mean_lead_time.toFixed(1)} steps
        </span>
      ),
    },
    {
      key: "actions",
      header: "Experiment",
      align: "right",
      sortable: false,
      render: (item) => (
        <Link
          to={`/experiments/exp_eval_${item.model_name.toLowerCase().replace(/[\s-]/g, "_")}`}
          className="text-[#3b82f6] hover:underline font-mono text-[11px]"
        >
          Inspect &rarr;
        </Link>
      ),
    },
  ];

  if (loading && comparisonModels.length === 0) {
    return <LoadingState message="Loading empirical model evaluation comparison..." />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={loadData} />;
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#242b3d] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#f0f3fa]">
            Canonical Model Evaluation & Leaderboard
          </h1>
          <p className="mt-1 text-xs text-[#9aa4bf]">
            Comparative empirical benchmarking across 9 architectures evaluated on AgentGuard dataset
          </p>
        </div>

        <div className="flex items-center gap-2">
          <ExportButton
            data={filteredModels}
            filename={`agentguard_model_comparison_k${horizon}`}
            format="csv"
            label="Export CSV"
          />
          <ExportButton
            data={filteredModels}
            filename={`agentguard_model_comparison_k${horizon}`}
            format="json"
            label="Export JSON"
          />
        </div>
      </div>

      {/* Filter Bar */}
      <FilterBar
        fields={[
          {
            key: "horizon",
            label: "Horizon (K)",
            value: String(horizon),
            options: [
              { label: "K = 1 step", value: "1" },
              { label: "K = 3 steps", value: "3" },
              { label: "K = 5 steps", value: "5" },
              { label: "K = 10 steps", value: "10" },
            ],
          },
          {
            key: "model",
            label: "Model",
            value: modelFilter,
            options: [
              { label: "Rule-Based", value: "Rule-Based" },
              { label: "Logistic Regression", value: "Logistic Regression" },
              { label: "Random Forest", value: "Random Forest" },
              { label: "XGBoost", value: "XGBoost" },
              { label: "LSTM", value: "LSTM" },
              { label: "GRU", value: "GRU" },
              { label: "GCN", value: "GCN" },
              { label: "GAT", value: "GAT" },
              { label: "Temporal GNN", value: "Temporal GNN" },
            ],
          },
        ]}
        onFilterChange={(key, val) => {
          if (key === "horizon") setHorizon(Number(val));
          if (key === "model") setModelFilter(val);
        }}
        onReset={() => {
          setHorizon(1);
          setModelFilter("");
        }}
      />

      {/* Primary Comparison DataTable */}
      <DataTable
        columns={columns}
        data={filteredModels}
        keyExtractor={(item) => item.model_name}
        searchableKey="model_name"
        searchPlaceholder="Filter models..."
        defaultSortKey="f1"
        defaultSortOrder="desc"
        pageSize={15}
        emptyMessage="No model results match the selected horizon or filter."
      />

      {/* Comparative Multi-Horizon Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* F1 Score vs Horizon */}
        <ChartContainer
          title="F1 Score vs Prediction Horizon"
          subtitle="Model predictive precision & recall balance as prediction horizon K increases"
          height={280}
        >
          <LineChart data={chartDataF1ByHorizon} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242b3d" />
            <XAxis dataKey="horizon" stroke="#5e6984" fontSize={11} />
            <YAxis stroke="#5e6984" fontSize={11} domain={[0, 1.05]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#141721", borderColor: "#333d56", fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            <Line type="monotone" dataKey="Rule-Based" stroke="#64748b" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Logistic Regression" stroke="#0ea5e9" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="LSTM" stroke="#f59e0b" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="GAT" stroke="#8b5cf6" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Temporal GNN" stroke="#3b82f6" strokeWidth={2.5} dot />
          </LineChart>
        </ChartContainer>

        {/* AUPRC vs Horizon */}
        <ChartContainer
          title="AUPRC vs Prediction Horizon"
          subtitle="Area Under Precision-Recall Curve under imbalanced cascading failure distributions"
          height={280}
        >
          <LineChart data={chartDataAuprcByHorizon} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242b3d" />
            <XAxis dataKey="horizon" stroke="#5e6984" fontSize={11} />
            <YAxis stroke="#5e6984" fontSize={11} domain={[0, 1.05]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#141721", borderColor: "#333d56", fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            <Line type="monotone" dataKey="Rule-Based" stroke="#64748b" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Logistic Regression" stroke="#0ea5e9" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="LSTM" stroke="#f59e0b" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="GAT" stroke="#8b5cf6" strokeWidth={1.5} dot />
            <Line type="monotone" dataKey="Temporal GNN" stroke="#3b82f6" strokeWidth={2.5} dot />
          </LineChart>
        </ChartContainer>

        {/* Lead Time vs Horizon */}
        <div className="lg:col-span-2">
          <ChartContainer
            title="Mean Early Warning Lead Time (Steps Before Cascade)"
            subtitle="Observed advance warning steps provided before cascading failure manifestation"
            height={260}
          >
            <BarChart
              data={chartDataLeadTimeByHorizon}
              margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#242b3d" />
              <XAxis dataKey="horizon" stroke="#5e6984" fontSize={11} />
              <YAxis stroke="#5e6984" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: "#141721", borderColor: "#333d56", fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              <Bar dataKey="Rule-Based" fill="#64748b" />
              <Bar dataKey="Logistic Regression" fill="#0ea5e9" />
              <Bar dataKey="Random Forest" fill="#10b981" />
              <Bar dataKey="LSTM" fill="#f59e0b" />
              <Bar dataKey="GAT" fill="#8b5cf6" />
              <Bar dataKey="Temporal GNN" fill="#3b82f6" />
            </BarChart>
          </ChartContainer>
        </div>
      </div>
    </div>
  );
};

export default ModelComparisonPage;
