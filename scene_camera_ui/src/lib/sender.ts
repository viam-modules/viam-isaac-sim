/**
 * Sends patches with at most one request in flight. Patches pushed while a request is pending
 * merge into one queued patch, so a fast drag costs one round trip at a time and the newest
 * values always win.
 */
export function createSender<P extends object, R>(
  send: (patch: P) => Promise<R>,
  onReply: (reply: R) => void,
  onError: (error: unknown) => void,
) {
  let queued: P | undefined;
  let inFlight = false;

  async function pump(): Promise<void> {
    if (inFlight || queued === undefined) return;
    const patch = queued;
    queued = undefined;
    inFlight = true;
    try {
      onReply(await send(patch));
    } catch (error) {
      onError(error);
    } finally {
      inFlight = false;
      void pump();
    }
  }

  return {
    push(patch: P) {
      queued = { ...queued, ...patch } as P;
      void pump();
    },
    /** Forget the queued patch, e.g. when the bridge disconnects. */
    drop() {
      queued = undefined;
    },
    get busy() {
      return inFlight || queued !== undefined;
    },
  };
}
