<script lang="ts">
  import type { CameraClient, JsonValue } from '@viamrobotics/sdk';
  import {
    createResourceClient,
    createResourceMutation,
    createResourceQuery,
    useBridge,
  } from '@viamrobotics/post-message-sdk/svelte';
  import { Badge, Button } from '@viamrobotics/prime-core';
  import { untrack } from 'svelte';

  import {
    RECORDING_ERROR,
    toView,
    type View,
    type ViewPatch,
    type ViewReply,
  } from './lib/contract';
  import { KEY_ORBIT_DEG, ORBIT_DEG_PER_PX, ZOOM_STEP, orbit, pan, zoom } from './lib/orbit';
  import { createSender } from './lib/sender';

  const FRAME_MS = 125; // ~8 fps
  const VIEW_POLL_MS = 2000; // clears the recording lock on its own

  const bridge = useBridge({ reportSize: true });
  const cam = createResourceClient<CameraClient>();
  const frame = createResourceQuery(cam, 'getImages', [], { refetchInterval: FRAME_MS });
  const viewQuery = createResourceQuery(cam, 'doCommand', [{ command: 'get_view' }], {
    refetchInterval: VIEW_POLL_MS,
  });
  const command = createResourceMutation(cam, 'doCommand');

  let local = $state<View | undefined>();
  let lastReply = $state<ViewReply | undefined>();
  let dragging = $state(false);
  let lockedByError = $state(false);
  let errorMessage = $state<string | undefined>();
  let src = $state<string | undefined>();

  const recording = $derived(lastReply?.recording === true || lockedByError);

  function adopt(reply: ViewReply) {
    lastReply = reply;
    if (!reply.recording) lockedByError = false;
    if (!dragging) local = toView(reply);
  }

  function messageOf(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function fail(error: unknown) {
    const message = messageOf(error);
    if (message.includes(RECORDING_ERROR)) {
      lockedByError = true;
      errorMessage = undefined;
    } else {
      errorMessage = message;
    }
    if (lastReply) local = toView(lastReply);
  }

  async function run(cmd: object): Promise<ViewReply> {
    return (await command.mutateAsync([cmd as Record<string, JsonValue>])) as unknown as ViewReply;
  }

  const sender = createSender<ViewPatch, ViewReply>(
    (patch) => run({ command: 'set_view', ...patch }),
    (reply) => {
      errorMessage = undefined;
      adopt(reply);
    },
    fail,
  );

  // server view: adopt each new poll unless the user is mid-interaction. Only new data re-runs
  // this; re-running when a drag ends would re-adopt the pre-drag poll and snap the view back.
  $effect(() => {
    const reply = viewQuery.data as unknown as ViewReply | undefined;
    untrack(() => {
      if (reply && !dragging && !sender.busy) adopt(reply);
    });
  });

  // a queued patch would be stale after a reconnect
  $effect(() => {
    if (bridge.status !== 'connected') sender.drop();
  });

  // frames: swap object URLs, revoking the old one; keep the last frame on errors
  let currentUrl: string | undefined;
  $effect(() => {
    // the camera serves one source; take the first image
    const image = frame.data?.images?.[0];
    if (!image || image.image.length === 0) return;
    const url = URL.createObjectURL(
      new Blob([image.image as Uint8Array<ArrayBuffer>], { type: image.mimeType }),
    );
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = url;
    src = url;
  });
  $effect(() => () => {
    if (currentUrl) URL.revokeObjectURL(currentUrl);
  });

  // prime-core's Button only sets aria-disabled; mirror it onto the native button so it really is
  let homeButton = $state<HTMLButtonElement | undefined>();
  $effect(() => {
    if (homeButton) homeButton.disabled = recording;
  });

  function update(next: View, patch: ViewPatch) {
    local = next;
    sender.push(patch);
  }

  async function goHome() {
    if (recording) return;
    sender.drop();
    try {
      adopt(await run({ command: 'reset_view' }));
      errorMessage = undefined;
    } catch (error) {
      fail(error);
    }
  }

  // ---- pointer input ----
  // plain Map on purpose: pointer positions drive math, not markup
  // eslint-disable-next-line svelte/prefer-svelte-reactivity
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDistance = 0;

  function spread(): number {
    const [a, b] = [...pointers.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }

  function onPointerDown(event: PointerEvent) {
    if (recording || !local) return;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    dragging = true;
    if (pointers.size === 2) pinchDistance = spread();
  }

  function onPointerMove(event: PointerEvent) {
    const last = pointers.get(event.pointerId);
    if (!dragging || !local || !last) return;
    const dx = event.clientX - last.x;
    const dy = event.clientY - last.y;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2) {
      const now = spread();
      if (pinchDistance > 0 && now > 0) {
        const next = zoom(local, pinchDistance / now);
        update(next, { distance: next.distance });
      }
      pinchDistance = now;
      // two fingers moving together also pan, at half rate per finger
      const next = pan(local, dx / 2, dy / 2);
      update(next, { target: next.target });
      return;
    }
    if (event.shiftKey) {
      const next = pan(local, dx, dy);
      update(next, { target: next.target });
    } else {
      const next = orbit(local, dx, dy);
      update(next, { yaw_deg: next.yaw_deg, pitch_deg: next.pitch_deg });
    }
  }

  function onPointerUp(event: PointerEvent) {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinchDistance = 0;
    if (pointers.size === 0) dragging = false;
  }

  function onWheel(event: WheelEvent) {
    event.preventDefault();
    if (recording || !local) return;
    const next = zoom(local, event.deltaY > 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
    update(next, { distance: next.distance });
  }

  function onKeyDown(event: KeyboardEvent) {
    if (recording || !local) return;
    const key = event.key;
    if (key === 'h' || key === 'H' || key === 'Home') {
      event.preventDefault();
      void goHome();
      return;
    }
    const px = KEY_ORBIT_DEG / ORBIT_DEG_PER_PX; // KEY_ORBIT_DEG expressed as drag pixels
    let next: View | undefined;
    if (key === 'ArrowLeft') next = orbit(local, -px, 0);
    else if (key === 'ArrowRight') next = orbit(local, px, 0);
    else if (key === 'ArrowUp') next = orbit(local, 0, px);
    else if (key === 'ArrowDown') next = orbit(local, 0, -px);
    else if (key === '+' || key === '=') next = zoom(local, 1 / ZOOM_STEP);
    else if (key === '-' || key === '_') next = zoom(local, ZOOM_STEP);
    if (!next) return;
    event.preventDefault();
    update(next, {
      yaw_deg: next.yaw_deg,
      pitch_deg: next.pitch_deg,
      distance: next.distance,
    });
  }

  let fovDraft = $state<number | undefined>();
  const fovShown = $derived(fovDraft ?? local?.fov_deg ?? 70);

  function commitFov() {
    if (fovDraft === undefined || !local) return;
    update({ ...local, fov_deg: fovDraft }, { fov_deg: fovDraft });
    fovDraft = undefined;
  }

  const fmt = (n: number, digits: number) => n.toFixed(digits);
</script>

<main class="flex flex-col gap-3 p-4 text-sm">
  <header class="flex items-center justify-between gap-2">
    <h1 class="text-base font-semibold">{cam.name ?? 'Scene camera'}</h1>
    {#if bridge.status === 'connected'}
      <Badge variant="success" label="Connected" />
    {:else if bridge.status === 'connecting'}
      <Badge variant="pending" label="Reconnecting" />
    {:else}
      <Badge variant="danger" label="Error" />
    {/if}
  </header>

  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
  <div
    class="relative aspect-[4/3] w-full touch-none overflow-hidden rounded bg-black outline-none select-none focus-visible:ring-2"
    class:cursor-grab={!recording && !dragging}
    class:cursor-grabbing={dragging}
    class:cursor-not-allowed={recording}
    tabindex="0"
    role="application"
    aria-label="Camera view. Drag to orbit, shift-drag to pan, scroll to zoom, double-click to return home. Arrow keys orbit, plus and minus zoom, H returns home."
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
    onwheel={onWheel}
    ondblclick={() => void goHome()}
    onkeydown={onKeyDown}
  >
    {#if src}
      <img
        {src}
        alt="Scene camera view"
        class="absolute inset-0 size-full object-contain"
        class:opacity-45={recording}
        draggable="false"
      />
    {:else}
      <p class="absolute inset-0 grid place-items-center text-subtle-2">
        Waiting for the first frame…
      </p>
    {/if}
    {#if dragging}
      <!-- the orbit target is always the image centre with a look-at camera -->
      <span
        class="pointer-events-none absolute top-1/2 left-1/2 size-5 -translate-1/2 rounded-full border-2 border-white"
        aria-hidden="true"
      ></span>
    {/if}
    <!-- the live region stays mounted so the badge appearing is announced -->
    <span class="absolute top-2 left-2" aria-live="polite">
      {#if recording}
        <Badge variant="warning" label="Recording episode: view locked" />
      {/if}
    </span>
  </div>

  {#if errorMessage}
    <p role="alert" class="text-danger-dark">{errorMessage}</p>
  {:else if frame.isError && src}
    <p role="status" class="text-subtle-2">Showing the last frame. {frame.error?.message}</p>
  {:else if frame.isError}
    <p role="alert" class="text-danger-dark">{frame.error?.message}</p>
  {/if}

  {#if local}
    <dl class="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <div>
        <dt class="text-xs text-subtle-2 uppercase">Distance</dt>
        <dd class="font-mono tabular-nums">{fmt(local.distance, 2)} m</dd>
      </div>
      <div>
        <dt class="text-xs text-subtle-2 uppercase">Yaw</dt>
        <dd class="font-mono tabular-nums">{fmt(local.yaw_deg, 1)}°</dd>
      </div>
      <div>
        <dt class="text-xs text-subtle-2 uppercase">Pitch</dt>
        <dd class="font-mono tabular-nums">{fmt(local.pitch_deg, 1)}°</dd>
      </div>
      <div>
        <dt class="text-xs text-subtle-2 uppercase">Target</dt>
        <dd class="font-mono tabular-nums">{local.target.map((n) => fmt(n, 2)).join(', ')}</dd>
      </div>
    </dl>
  {/if}

  <div class="flex flex-wrap items-center gap-3">
    <Button
      bind:button={homeButton}
      variant="primary"
      disabled={recording}
      onclick={() => void goHome()}>Home</Button
    >
    <label class="flex min-w-48 flex-1 items-center gap-2" for="fov">
      <span class="text-xs text-subtle-2 uppercase">FOV</span>
      <input
        id="fov"
        type="range"
        min="20"
        max="120"
        step="1"
        class="flex-1"
        value={fovShown}
        disabled={recording || !local}
        oninput={(e) => (fovDraft = Number((e.currentTarget as HTMLInputElement).value))}
        onchange={commitFov}
      />
      <span class="w-10 font-mono tabular-nums">{Math.round(fovShown)}°</span>
    </label>
  </div>
</main>
