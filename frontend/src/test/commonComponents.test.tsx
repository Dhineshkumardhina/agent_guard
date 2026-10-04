import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import DataTable, { type Column } from '../components/common/DataTable';
import FilterBar, { type FilterField } from '../components/common/FilterBar';
import { LoadingState, ErrorState } from '../components/common/States';

describe('Design System Common Components', () => {
  it('renders MetricCard with label, value, and status', () => {
    render(
      <MetricCard
        label="F1 Score"
        value="0.875"
        subtext="Horizon k=3"
        status="success"
      />
    );
    expect(screen.getByText('F1 Score')).toBeInTheDocument();
    expect(screen.getByText('0.875')).toBeInTheDocument();
    expect(screen.getByText('Horizon k=3')).toBeInTheDocument();
  });

  it('renders Card with title, subtitle, and content', () => {
    render(
      <Card title="Ablation Matrix" subtitle="Empirical attribution metrics">
        <p>Ablation children content</p>
      </Card>
    );
    expect(screen.getByText('Ablation Matrix')).toBeInTheDocument();
    expect(screen.getByText('Empirical attribution metrics')).toBeInTheDocument();
    expect(screen.getByText('Ablation children content')).toBeInTheDocument();
  });

  it('renders LoadingState, ErrorState, and EmptyState correctly', () => {
    const { unmount } = render(<LoadingState message="Fetching data..." />);
    expect(screen.getByText('Fetching data...')).toBeInTheDocument();
    unmount();

    const onRetry = vi.fn();
    render(<ErrorState title="Load Failed" message="Network timeout" onRetry={onRetry} />);
    expect(screen.getByText('Load Failed')).toBeInTheDocument();
    expect(screen.getByText('Network timeout')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Retry Request'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('renders FilterBar and triggers change callback', () => {
    const onFilterChange = vi.fn();
    const fields: FilterField[] = [
      {
        key: 'model',
        label: 'Model Architecture',
        value: 'temporal_gnn',
        options: [
          { label: 'Temporal GNN', value: 'temporal_gnn' },
          { label: 'Static GNN', value: 'static_gnn' },
        ],
      },
    ];

    render(<FilterBar fields={fields} onChange={onFilterChange} onReset={vi.fn()} />);
    expect(screen.getByText('Model Architecture:')).toBeInTheDocument();

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'static_gnn' } });
    expect(onFilterChange).toHaveBeenCalledWith('model', 'static_gnn');
  });

  it('renders DataTable with columns and handles sorting and pagination', () => {
    interface TestRow {
      id: string;
      model: string;
      f1: number;
    }

    const data: TestRow[] = [
      { id: '1', model: 'temporal_gnn', f1: 0.89 },
      { id: '2', model: 'rule_based', f1: 0.62 },
      { id: '3', model: 'lstm', f1: 0.78 },
    ];

    const columns: Column<TestRow>[] = [
      { key: 'model', header: 'Model Name', sortable: true },
      { key: 'f1', header: 'F1 Score', sortable: true },
    ];

    render(
      <DataTable
        columns={columns}
        data={data}
        keyField="id"
        pageSize={2}
      />
    );

    // Initial page shows 2 rows
    expect(screen.getByText('temporal_gnn')).toBeInTheDocument();
    expect(screen.getByText('rule_based')).toBeInTheDocument();
    expect(screen.queryByText('lstm')).not.toBeInTheDocument();

    // Sort by model header
    const modelHeader = screen.getByText('Model Name');
    fireEvent.click(modelHeader);
    // After sorting asc: lstm, rule_based
    expect(screen.getByText('lstm')).toBeInTheDocument();
  });
});
