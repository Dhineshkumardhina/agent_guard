import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getExperimentById } from '../api/experiments';
import { getEvaluationById } from '../api/evaluations';
import { getPredictions } from '../api/predictions';
import type { ExperimentResponse, EvaluationResponse, PredictionResponse } from '../types';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { StatusBadge, ModelBadge, RiskBadge } from '../components/common/Badges';
import { LoadingState, ErrorState, EmptyState } from '../components/common/States';
import DataTable, { type Column } from '../components/common/DataTable';
import ExportButton from '../components/common/ExportButton';
import { ArrowLeft, Cpu, Database, Hash, Calendar, Settings, Activity } from 'lucide-react';

export const ExperimentDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [experiment, setExperiment] = useState<ExperimentResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    Promise.all([
      getExperimentById(id),
      getEvaluationById(id).catch(() => null),
      getPredictions({ limit: 100 }).catch(() => ({ items: [], total: 0 })),
    ])
      .then(([expData, evalData, predRes]) => {
        if (!isMounted) return;
        const predItems = (predRes as any).items || [];
        setExperiment(expData);
        setEvaluation(evalData);
        // Filter predictions that might match this experiment/model
        if (expData && expData.model) {
          const matched = predItems.filter((p: PredictionResponse) => p.model === expData.model);
          setPredictions(matched.length > 0 ? matched : predItems);
        } else {
          setPredictions(predItems);
        }
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setError(err.message || 'Failed to load experiment details.');
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [id]);

  if (loading) {
    return <LoadingState message="Loading experiment metadata and evaluation records..." />;
  }

  if (error || !experiment) {
    return (
      <ErrorState
        title="Experiment Not Found"
        message={error || `Experiment '${id}' could not be located in the database.`}
        onRetry={() => window.location.reload()}
      />
    );
  }

  const predictionColumns: Column<PredictionResponse>[] = [
    {
      header: 'Prediction ID',
      accessor: (p) => (
        <span className="font-mono text-xs text-slate-300" title={p.prediction_id}>
          {p.prediction_id.slice(0, 16)}...
        </span>
      ),
    },
    {
      header: 'Run ID',
      accessor: (p) => (
        <Link
          to={`/runs/${p.run_id}`}
          className="font-mono text-xs text-cyan-400 hover:text-cyan-300 hover:underline"
        >
          {p.run_id.slice(0, 14)}
        </Link>
      ),
    },
    {
      header: 'Step',
      accessor: (p) => p.step_idx ?? '-',
    },
    {
      header: 'Horizon',
      accessor: (p) => <span className="font-mono text-xs">k={p.horizon}</span>,
    },
    {
      header: 'Probability',
      accessor: (p) => (
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${
                p.predicted_probability >= (p.threshold ?? 0.5) ? 'bg-amber-400' : 'bg-emerald-400'
              }`}
              style={{ width: `${Math.min(100, Math.round(p.predicted_probability * 100))}%` }}
            />
          </div>
          <span className="font-mono text-xs text-slate-200">
            {p.predicted_probability.toFixed(3)}
          </span>
        </div>
      ),
      sortable: true,
      sortBy: (p) => p.predicted_probability,
    },
    {
      header: 'Threshold',
      accessor: (p) => <span className="font-mono text-xs text-slate-400">τ={p.threshold?.toFixed(2) ?? '0.50'}</span>,
    },
    {
      header: 'Risk Level',
      accessor: (p) => <RiskBadge risk={p.risk_level || (p.predicted_label === 1 ? 'HIGH' : 'LOW')} />,
    },
    {
      header: 'Actual Outcome',
      accessor: (p) => {
        if (p.actual_outcome === undefined) return <span className="text-slate-500 text-xs">Pending</span>;
        return p.actual_outcome === 1 ? (
          <span className="text-xs px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40">
            Failure
          </span>
        ) : (
          <span className="text-xs px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
            Normal
          </span>
        );
      },
    },
    {
      header: 'Lead Time',
      accessor: (p) =>
        p.lead_time !== undefined && p.lead_time !== null ? (
          <span className="font-mono text-xs text-slate-300">{p.lead_time} steps</span>
        ) : (
          <span className="text-slate-600 text-xs">-</span>
        ),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Action */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            to="/experiments"
            className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100 tracking-tight">{experiment.name}</h1>
              <StatusBadge status={experiment.status} />
              {experiment.model && <ModelBadge model={experiment.model} />}
            </div>
            <p className="text-xs font-mono text-slate-500 mt-0.5">ID: {experiment.experiment_id}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ExportButton data={experiment} filename={`experiment_${experiment.experiment_id}`} label="Export Config" />
          {evaluation && (
            <ExportButton
              data={evaluation}
              filename={`evaluation_${experiment.experiment_id}`}
              label="Export Evaluation"
            />
          )}
        </div>
      </div>

      {/* Metadata Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="flex items-center gap-3">
          <div className="p-2 rounded bg-slate-900 text-cyan-400 border border-slate-800">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs text-slate-500">Model Architecture</div>
            <div className="text-sm font-semibold text-slate-200">{experiment.model || 'Unknown'}</div>
          </div>
        </Card>
        <Card className="flex items-center gap-3">
          <div className="p-2 rounded bg-slate-900 text-purple-400 border border-slate-800">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs text-slate-500">Dataset Version</div>
            <div className="text-sm font-semibold text-slate-200">{experiment.dataset_version || 'v1.0'}</div>
          </div>
        </Card>
        <Card className="flex items-center gap-3">
          <div className="p-2 rounded bg-slate-900 text-amber-400 border border-slate-800">
            <Hash className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs text-slate-500">Random Seed</div>
            <div className="text-sm font-mono font-semibold text-slate-200">{experiment.seed}</div>
          </div>
        </Card>
        <Card className="flex items-center gap-3">
          <div className="p-2 rounded bg-slate-900 text-emerald-400 border border-slate-800">
            <Calendar className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs text-slate-500">Created At</div>
            <div className="text-xs font-mono text-slate-300">
              {new Date(experiment.creation_time).toLocaleDateString()}
            </div>
          </div>
        </Card>
      </div>

      {/* Evaluation Metrics (if available) */}
      {evaluation ? (
        <Card
          title="Empirical Evaluation Performance"
          subtitle={`Horizon k=${evaluation.horizon} · Sample size: ${evaluation.sample_count ?? 'N/A'}`}
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-4">
            <MetricCard
              label="F1 Score"
              value={evaluation.f1.toFixed(3)}
              status={evaluation.f1 > 0.8 ? 'success' : evaluation.f1 > 0.6 ? 'warning' : 'danger'}
            />
            <MetricCard
              label="AUROC"
              value={evaluation.auroc.toFixed(3)}
              status={evaluation.auroc > 0.8 ? 'success' : 'neutral'}
            />
            <MetricCard
              label="AUPRC"
              value={evaluation.auprc.toFixed(3)}
              status={evaluation.auprc > 0.75 ? 'success' : 'neutral'}
            />
            <MetricCard
              label="Precision"
              value={evaluation.precision.toFixed(3)}
            />
            <MetricCard
              label="Recall"
              value={evaluation.recall.toFixed(3)}
            />
            <MetricCard
              label="False Alarm Rate"
              value={`${(evaluation.false_alarm_rate * 100).toFixed(1)}%`}
              status={evaluation.false_alarm_rate < 0.1 ? 'success' : 'danger'}
            />
            <MetricCard
              label="Mean Lead Time"
              value={`${evaluation.mean_lead_time.toFixed(1)} steps`}
              status="info"
            />
          </div>

          <div className="text-xs text-slate-400 bg-slate-950 p-3 rounded border border-slate-800 flex flex-wrap gap-x-6 gap-y-2">
            <span>
              <strong className="text-slate-300">Median Lead Time:</strong> {evaluation.median_lead_time} steps
            </span>
            <span>
              <strong className="text-slate-300">Successful Warnings:</strong> {evaluation.successful_warnings}
            </span>
            <span>
              <strong className="text-slate-300">Warnings/Trajectory:</strong> {evaluation.warnings_per_trajectory}
            </span>
            {evaluation.brier_score !== undefined && (
              <span>
                <strong className="text-slate-300">Brier Score:</strong> {evaluation.brier_score.toFixed(4)}
              </span>
            )}
            {evaluation.ece !== undefined && (
              <span>
                <strong className="text-slate-300">ECE (Calibration):</strong> {evaluation.ece.toFixed(4)}
              </span>
            )}
          </div>
        </Card>
      ) : (
        <Card title="Empirical Evaluation Performance">
          <EmptyState
            title="No Evaluation Run Recorded"
            message="No standalone evaluation record is associated with this experiment ID. Run an evaluation task to populate validation metrics."
          />
        </Card>
      )}

      {/* Configuration & Hyperparameters */}
      <Card
        title="Experiment Configuration & Hyperparameters"
        subtitle="Exact configuration parameters used during training and inference"
        icon={<Settings className="w-4 h-4 text-cyan-400" />}
      >
        {Object.keys(experiment.configuration || {}).length > 0 ? (
          <div className="bg-slate-950 border border-slate-800 rounded p-4 overflow-x-auto">
            <pre className="text-xs font-mono text-cyan-300">
              {JSON.stringify(experiment.configuration, null, 2)}
            </pre>
          </div>
        ) : (
          <div className="text-xs text-slate-500 italic p-3">No hyperparameter overrides recorded.</div>
        )}
      </Card>

      {/* Predictions Sample Table */}
      <Card
        title="Associated Inference Predictions"
        subtitle="Individual prediction events generated by this model on simulation trajectories"
        icon={<Activity className="w-4 h-4 text-amber-400" />}
        action={
          predictions.length > 0 ? (
            <ExportButton data={predictions} filename={`predictions_${experiment.experiment_id}`} label="Export Predictions" />
          ) : undefined
        }
      >
        <DataTable
          columns={predictionColumns}
          data={predictions}
          keyField="prediction_id"
          emptyMessage="No individual predictions logged for this experiment."
          pageSize={10}
          searchPlaceholder="Search predictions by Run ID..."
        />
      </Card>
    </div>
  );
};

export default ExperimentDetailPage;
