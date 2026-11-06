<script lang="ts">
  import type { SensorClient } from '@viamrobotics/sdk';
  import {
    createResourceClient,
    createResourceQuery,
    useBridge,
  } from '@viamrobotics/post-message-sdk/svelte';
  import { Badge, Button } from '@viamrobotics/prime-core';

  const POLL_MS = 1000;

  const bridge = useBridge();
  const sensor = createResourceClient<SensorClient>();
  const readings = createResourceQuery(sensor, 'getReadings', { refetchInterval: POLL_MS });

  const entries = $derived(Object.entries(readings.data ?? {}));
  // Once readings exist they stay on screen. Later polls, reconnects and failures only change the
  // indicator and the notice, so the control does not flash on every refetch.
  const hasReadings = $derived(readings.data !== undefined);
  const isBusy = $derived(readings.isFetching || bridge.status === 'connecting');
  const staleReason = $derived.by(() => {
    if (bridge.status === 'connecting') return 'Reconnecting to the machine.';
    if (bridge.status === 'error') {
      return `Could not connect: ${bridge.error?.message ?? 'unknown error'}`;
    }
    if (readings.isError) {
      return `Could not read the sensor: ${readings.error?.message ?? 'unknown error'}`;
    }
    return undefined;
  });
</script>

<main class="flex flex-col gap-3 p-4 text-sm">
  <header class="flex items-center justify-between gap-2">
    <h1 class="text-base font-semibold">{sensor.name ?? 'Sensor'}</h1>
    <div class="flex items-center gap-2">
      <span
        class="size-2 rounded-full bg-current text-subtle-2 transition-opacity motion-safe:animate-pulse"
        class:opacity-0={!isBusy}
        aria-hidden="true"
      ></span>
      {#if bridge.status === 'connected'}
        <Badge variant="success" label="Connected" />
      {:else if bridge.status === 'connecting'}
        <Badge variant="pending" label="Connecting" />
      {:else}
        <Badge variant="danger" label="Error" />
      {/if}
    </div>
  </header>

  {#if !hasReadings}
    {#if bridge.status === 'connecting'}
      <p role="status" class="text-subtle-2">Connecting to the machine…</p>
    {:else if bridge.status === 'error'}
      <p role="alert" class="text-danger-dark">
        Could not connect: {bridge.error?.message ?? 'unknown error'}
      </p>
    {:else if readings.isError}
      <p role="alert" class="text-danger-dark">
        Could not read the sensor: {readings.error?.message ?? 'unknown error'}
      </p>
    {:else}
      <p role="status" class="text-subtle-2">Loading readings…</p>
    {/if}
  {:else}
    {#if staleReason !== undefined}
      <p role="status" class="text-subtle-2">Showing the last readings. {staleReason}</p>
    {/if}
    {#if entries.length === 0}
      <p class="text-subtle-2">This sensor returned no readings.</p>
    {:else}
      <ul class="divide-light divide-y border border-light">
        {#each entries as [key, value] (key)}
          <li class="flex justify-between gap-4 px-3 py-2">
            <span class="font-medium">{key}</span>
            <span class="font-mono">{JSON.stringify(value)}</span>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}

  <div>
    <Button variant="ghost-outline" onclick={() => void readings.refetch()}>Refetch</Button>
  </div>
</main>
