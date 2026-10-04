import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';

describe('App Routing and Views', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/health')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve({ status: 'ok', service: 'agentguard-api' }),
          });
        }
        if (url.includes('/models/comparison') || url.includes('/evaluations/comparison')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () =>
              Promise.resolve({
                horizon: 1,
                total_models: 2,
                models: [
                  {
                    model_name: 'temporal_gnn',
                    model_family: 'Temporal GNN',
                    horizon: 1,
                    f1: 0.89,
                    auroc: 0.94,
                    auprc: 0.88,
                    precision: 0.86,
                    recall: 0.92,
                    false_alarm_rate: 0.05,
                    mean_lead_time: 4.2,
                    median_lead_time: 4.0,
                    successful_warnings: 32,
                    warnings_per_trajectory: 1.1,
                    normalized_f1: 1.0,
                    normalized_auroc: 1.0,
                    normalized_lead_time: 1.0,
                  },
                ],
                best_model_by_f1: 'temporal_gnn',
                best_model_by_auroc: 'temporal_gnn',
                best_model_by_lead_time: 'temporal_gnn',
              }),
          });
        }
        if (url.includes('/runs')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () =>
              Promise.resolve({
                items: [
                  {
                    id: 'run_test_001',
                    task: 'collaborative_coding',
                    topology: 'pipeline',
                    status: 'completed',
                    agent_count: 5,
                    duration_seconds: 14.5,
                    has_cascading_failure: false,
                    random_seed: 42,
                    created_at: new Date().toISOString(),
                  },
                ],
                total: 1,
                limit: 50,
                offset: 0,
                has_more: false,
              }),
          });
        }
        if (url.includes('/experiments')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () =>
              Promise.resolve({
                items: [],
                total: 0,
                limit: 50,
                offset: 0,
                has_more: false,
              }),
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ items: [], total: 0 }),
        });
      })
    );
  });

  it('renders application layout with navigation sidebar and header', async () => {
    render(<App />);

    expect(screen.getByText('AgentGuard')).toBeInTheDocument();
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('Model Comparison')).toBeInTheDocument();
    expect(screen.getByText('Simulation Runs')).toBeInTheDocument();
    expect(screen.getByText('Early Warning')).toBeInTheDocument();
    expect(screen.getByText('Ablations')).toBeInTheDocument();
    expect(screen.getByText('Generalization')).toBeInTheDocument();
    expect(screen.getByText('Explainability')).toBeInTheDocument();
  });
});
