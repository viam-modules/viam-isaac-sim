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

// jsdom has no PointerEvent or pointer capture
class TestPointerEvent extends MouseEvent {
  pointerId: number;
  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
  }
}
Object.assign(window, { PointerEvent: TestPointerEvent });
HTMLElement.prototype.setPointerCapture = () => undefined;

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

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

  function drag(dx: number) {
    const view = screen.getByRole('application');
    view.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        button: 0,
        clientX: 0,
        clientY: 0,
        bubbles: true,
      }),
    );
    view.dispatchEvent(
      new PointerEvent('pointermove', { pointerId: 1, clientX: dx, clientY: 0, bubbles: true }),
    );
    view.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1, bubbles: true }));
  }

  it('keeps the dragged view when the bridge drops the request', async () => {
    const lost = Object.assign(new Error('The bridge is not connected yet.'), {
      kind: 'not-connected',
    });
    state.mutateAsync = vi.fn().mockRejectedValue(lost);
    render(App);
    drag(100); // 0.3 deg/px: yaw 0 -> -30
    await flush();
    expect(screen.getByText('-30.0°')).toBeTruthy();
  });

  it('sends a payload postMessage can clone (no reactive proxies)', async () => {
    render(App);
    drag(100); // orbit keeps the existing target array
    await flush();
    const [args] = state.mutateAsync.mock.calls.at(-1) as [unknown];
    expect(() => structuredClone(args)).not.toThrow();
  });

  it('surfaces a request the bridge could not post instead of retrying it silently', async () => {
    const unpostable = Object.assign(new Error('Failed to post request to the host.'), {
      kind: 'not-connected',
      cause: new Error('could not be cloned'),
    });
    state.mutateAsync = vi.fn().mockRejectedValue(unpostable);
    render(App);
    drag(100);
    await flush();
    expect(screen.getByRole('alert').textContent).toContain('Failed to post');
  });

  it('sends the whole view, so a later request heals an earlier lost one', async () => {
    render(App);
    drag(100);
    await flush();
    expect(state.mutateAsync).toHaveBeenLastCalledWith([
      {
        command: 'set_view',
        target: [0, 0, 0.75],
        distance: 3,
        yaw_deg: -30,
        pitch_deg: 20,
        fov_deg: 70,
      },
    ]);
  });

  it('rolls back when the machine rejects the view', async () => {
    state.mutateAsync = vi.fn().mockRejectedValue(new Error('set_view: target must be [x, y, z]'));
    render(App);
    drag(100);
    await flush();
    expect(screen.getByText('0.0°')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toContain('target must be');
  });
});
