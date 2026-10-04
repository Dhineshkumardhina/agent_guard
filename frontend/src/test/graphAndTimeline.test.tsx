import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import GraphViewer from '../components/graphs/GraphViewer';
import FailureTimeline from '../components/timeline/FailureTimeline';
import type { RunGraphResponse, EventResponse, RunFailureResponse, PredictionResponse } from '../types';

describe('Graph and Timeline Visualizations', () => {
  const mockGraph: RunGraphResponse = {
    run_id: 'run_001',
    nodes: [
      { id: 'agent_0', role: 'Planner' },
      { id: 'agent_1', role: 'Coder' },
      { id: 'agent_2', role: 'Verifier' },
    ],
    edges: [
      { source: 'agent_0', target: 'agent_1', interaction_count: 5 },
      { source: 'agent_1', target: 'agent_2', interaction_count: 3 },
    ],
    temporal_snapshots: [
      {
        snapshot_idx: 0,
        timestamp: 1.5,
        step_idx: 1,
        num_nodes: 3,
        num_edges: 2,
        nodes: [
          { id: 'agent_0', role: 'Planner' },
          { id: 'agent_1', role: 'Coder' },
          { id: 'agent_2', role: 'Verifier' },
        ],
        edges: [
          { source: 'agent_0', target: 'agent_1' },
          { source: 'agent_1', target: 'agent_2' },
        ],
      },
    ],
    density: 0.667,
    diameter: 2,
    average_clustering: 0.0,
    is_connected: true,
  };

  it('renders GraphViewer with agents, edges, and density statistics', () => {
    render(<GraphViewer graphData={mockGraph} />);

    expect(screen.getByText('Temporal Graph Snapshot:')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument(); // 3 agents
    expect(screen.getByText('2')).toBeInTheDocument(); // 2 edges
    expect(screen.getByText('0.667')).toBeInTheDocument(); // density
    expect(screen.getByText('agent_0')).toBeInTheDocument();
    expect(screen.getByText('agent_1')).toBeInTheDocument();
    expect(screen.getByText('agent_2')).toBeInTheDocument();
  });

  it('renders FailureTimeline and calculates lead time correctly', () => {
    const events: EventResponse[] = [
      {
        id: 'ev_1',
        run_id: 'run_001',
        step_idx: 1,
        timestamp: 1.0,
        source_agent: 'agent_0',
        target_agent: 'agent_1',
        event_type: 'task_request',
        message_length: 50,
        token_count: 12,
        latency: 0.1,
        confidence: 0.95,
        output_quality: 0.9,
        contradiction_score: 0.0,
        tool_error: false,
        retry_count: 0,
        failure_label: 0,
        downstream_failure: false,
        metadata_json: {},
        created_at: new Date().toISOString(),
      },
    ];

    const failures: RunFailureResponse[] = [
      {
        id: 'fail_1',
        run_id: 'run_001',
        step_idx: 8,
        failure_level: 2,
        originating_agent: 'agent_1',
        affected_agents: ['agent_2'],
        failure_type: 'semantic_drift',
        created_at: new Date().toISOString(),
      },
    ];

    const predictions: PredictionResponse[] = [
      {
        prediction_id: 'pred_1',
        run_id: 'run_001',
        model: 'temporal_gnn',
        timestamp: 4.0,
        step_idx: 4,
        horizon: 3,
        predicted_probability: 0.85,
        predicted_label: 1,
        threshold: 0.5,
      },
    ];

    render(
      <FailureTimeline
        events={events}
        failures={failures}
        predictions={predictions}
        cascadeStep={8}
        totalSteps={10}
      />
    );

    // Warning raised at Step 4, cascade at Step 8 -> Lead time = +4 steps
    expect(screen.getByText(/Warning Raised: Step 4/i)).toBeInTheDocument();
    expect(screen.getByText(/Cascade Triggered: Step 8/i)).toBeInTheDocument();
    expect(screen.getByText(/Advance Lead Time: \+4 steps/i)).toBeInTheDocument();
  });
});
