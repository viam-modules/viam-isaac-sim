import { render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

type BridgeStatus = 'connecting' | 'connected' | 'error';

const state = vi.hoisted(() => ({
  bridge: {
    status: 'connected' as BridgeStatus,
    resourceName: 'my-sensor',
    error: undefined as { message: string } | undefined,
  },
  query: {
    data: undefined as Record<string, unknown> | undefined,
    error: undefined as { message: string } | undefined,
    isLoading: false,
    isFetching: false,
    isError: false,
    refetch: () => Promise.resolve(),
  },
}));

vi.mock('@viamrobotics/post-message-sdk/svelte', () => ({
  useBridge: () => state.bridge,
  createResourceClient: () => ({ current: {}, name: 'my-sensor' }),
  createResourceQuery: () => state.query,
}));

import App from './App.svelte';

describe('App', () => {
  beforeEach(() => {
    state.bridge.status = 'connected';
    state.bridge.error = undefined;
    state.query.data = undefined;
    state.query.error = undefined;
    state.query.isLoading = false;
    state.query.isFetching = false;
    state.query.isError = false;
  });

  it('renders the readings', () => {
    state.query.data = { temperature: 21.5, humidity: 40 };
    render(App);
    expect(screen.getByText('temperature')).toBeTruthy();
    expect(screen.getByText('21.5')).toBeTruthy();
    expect(screen.getByText('humidity')).toBeTruthy();
  });

  it('keeps the readings on screen while a poll and a reconnect are in flight', () => {
    state.query.data = { temperature: 21.5 };
    state.query.isFetching = true;
    state.bridge.status = 'connecting';
    render(App);
    expect(screen.getByText('temperature')).toBeTruthy();
    expect(screen.queryByText('Connecting to the machine…')).toBeNull();
    expect(screen.getByRole('status').textContent).toContain('Showing the last readings');
  });

  it('keeps the readings with a notice when a later poll fails', () => {
    state.query.data = { temperature: 21.5 };
    state.query.isError = true;
    state.query.error = { message: 'boom' };
    render(App);
    expect(screen.getByText('temperature')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('status').textContent).toContain('boom');
  });

  it('renders the loading placeholder only before the first readings', () => {
    state.query.isLoading = true;
    state.query.isFetching = true;
    render(App);
    expect(screen.getByRole('status').textContent).toContain('Loading readings…');
  });

  it('renders the error state when the query fails before any readings', () => {
    state.query.isError = true;
    state.query.error = { message: 'boom' };
    render(App);
    expect(screen.getByRole('alert').textContent).toContain('boom');
  });
});
