import { describe, it, expect, vi } from 'vitest';
import { buildQueryString, apiFetch, ApiError } from '../api/client';

describe('API Client Utility Tests', () => {
  it('correctly serializes query string parameters', () => {
    const params = {
      task: 'collaborative_coding',
      limit: 10,
      has_cascade: true,
      empty_param: '',
      null_param: undefined as any,
    };
    const qs = buildQueryString(params);
    expect(qs).toBe('?task=collaborative_coding&limit=10&has_cascade=true');
  });

  it('returns empty string when no valid parameters exist', () => {
    expect(buildQueryString({})).toBe('');
    expect(buildQueryString({ a: undefined as any })).toBe('');
  });

  it('throws ApiError on HTTP non-200 responses', async () => {
    const mockResponse = {
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: () => Promise.resolve({ detail: 'Run not found in database' }),
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

    await expect(apiFetch('/api/v1/runs/invalid_run')).rejects.toThrow(ApiError);
    await expect(apiFetch('/api/v1/runs/invalid_run')).rejects.toThrow('Run not found in database');
  });

  it('correctly parses JSON on successful responses', async () => {
    const mockData = { id: 'run_123', status: 'completed' };
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockData),
      })
    );

    const result = await apiFetch<typeof mockData>('/api/v1/runs/run_123');
    expect(result).toEqual(mockData);
  });
});
