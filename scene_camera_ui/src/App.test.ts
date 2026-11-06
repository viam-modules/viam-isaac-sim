import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const state = vi.hoisted(() => ({
  view: undefined as unknown as Record<string, unknown>,
  mutateAsync: undefined as unknown as ReturnType<typeof vi.fn>,
}));

vi.mock('@viamrobotics/post-message-sdk/svelte', () => {
  const query = (data: () => unknown) => ({
    get data() {
      return data();
    },
    error: undefined,
    isLoading: false,
    isFetching: false,
    isError: false,
    isSuccess: true,
    refetch: () => Promise.resolve(),
  });
  return {
    useBridge: () => ({ status: 'connected', resourceName: 'overview', error: undefined }),
    createResourceClient: () => ({ current: {}, name: 'overview' }),
    createResourceQuery: (_client: unknown, method: string) =>
      method === 'getImages' ? query(() => undefined) : query(() => state.view),
    createResourceMutation: () => ({
      get mutateAsync() {
        return state.mutateAsync;
      },
      isPending: false,
      isError: false,
      error: undefined,
    }),
  };
});

import App from './App.svelte';

const home = { target: [0, 0, 0.75], distance: 3, yaw_deg: 0, pitch_deg: 20, fov_deg: 70 };

describe('App', () => {
  beforeEach(() => {
    state.view = { ...home, home, recording: false };
    state.mutateAsync = vi.fn().mockResolvedValue({ ...home, home, recording: false });
  });

  it('shows the view readout', () => {
    render(App);
    expect(screen.getByText('3.00 m')).toBeTruthy();
    expect(screen.getByText('20.0°')).toBeTruthy();
    expect(screen.getByText('0.00, 0.00, 0.75')).toBeTruthy();
  });

  it('locks the controls while an episode records', () => {
    state.view = { ...home, home, recording: true };
    render(App);
    expect(screen.getByText('Recording episode: view locked')).toBeTruthy();
    expect((screen.getByRole('button', { name: 'Home' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('Home sends reset_view', async () => {
    render(App);
    await fireEvent.click(screen.getByRole('button', { name: 'Home' }));
    expect(state.mutateAsync).toHaveBeenCalledWith([{ command: 'reset_view' }]);
  });
});
