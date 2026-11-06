/**
 * Sends patches with at most one request in flight. Patches pushed while a request is pending
 * merge into one queued patch, so a fast drag costs one round trip at a time and the newest
 * values always win. A reply is reported only when nothing is queued behind it; errors always are.
 *
 * When `onError` returns true the failed patch is kept (unless a newer one is queued) and sending
 * stalls until the next `push` or `resume`, so a dropped connection never spins in a retry loop
 * and never loses the latest patch.
 */
export function createSender<P extends object, R>(
  send: (patch: P) => Promise<R>,
  onReply: (reply: R) => void,
  onError: (error: unknown) => boolean | void,
) {
  let queued: P | undefined;
  let inFlight = false;
  let stalled = false;

  async function pump(): Promise<void> {
    if (inFlight || stalled || queued === undefined) return;
    const patch = queued;
    queued = undefined;
    inFlight = true;
    try {
      const reply = await send(patch);
      // a queued patch supersedes this reply; reporting it would roll the caller's view back
      if (queued === undefined) onReply(reply);
    } catch (error) {
      if (onError(error) === true) {
        queued ??= patch;
        stalled = true;
      }
    } finally {
      inFlight = false;
      void pump();
    }
  }

  return {
    push(patch: P) {
      queued = { ...queued, ...patch } as P;
      stalled = false;
      void pump();
    },
    /** Retry a stalled patch, e.g. once the bridge reconnects. */
    resume() {
      stalled = false;
      void pump();
    },
    /** Forget the queued patch, e.g. when Home supersedes it. */
    drop() {
      queued = undefined;
    },
    get busy() {
      return inFlight || queued !== undefined;
    },
  };
}
