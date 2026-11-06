import { describe, expect, it, vi } from 'vitest';

import { createSender } from './sender';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('createSender retry', () => {
  it('keeps a patch after a retryable error and resends it on resume', async () => {
    const send = vi.fn().mockRejectedValueOnce(new Error('lost')).mockResolvedValueOnce('ok');
    const onReply = vi.fn();
    const sender = createSender(send, onReply, () => true);
    sender.push({ a: 1 });
    await flush();
    expect(send).toHaveBeenCalledTimes(1); // no tight retry loop
    expect(sender.busy).toBe(true);
    sender.resume();
    await flush();
    expect(send).toHaveBeenLastCalledWith({ a: 1 });
    expect(onReply).toHaveBeenCalledWith('ok');
    expect(sender.busy).toBe(false);
  });

  it('a newer push replaces the stalled patch and sends at once', async () => {
    const send = vi.fn().mockRejectedValueOnce(new Error('lost')).mockResolvedValueOnce('ok');
    const sender = createSender(send, vi.fn(), () => true);
    sender.push({ a: 1 });
    await flush();
    sender.push({ a: 2 });
    await flush();
    expect(send).toHaveBeenLastCalledWith({ a: 2 });
    expect(send).toHaveBeenCalledTimes(2);
  });

  it('does not keep a patch the error handler declines to retry', async () => {
    const send = vi.fn().mockRejectedValueOnce(new Error('rejected'));
    const sender = createSender(send, vi.fn(), () => false);
    sender.push({ a: 1 });
    await flush();
    expect(sender.busy).toBe(false);
  });
});

describe('createSender', () => {
  it('keeps one request in flight and sends only the newest merged patch next', async () => {
    const calls: Record<string, number>[] = [];
    const replies: ReturnType<typeof deferred<string>>[] = [];
    const send = vi.fn((patch: Record<string, number>) => {
      calls.push(patch);
      const d = deferred<string>();
      replies.push(d);
      return d.promise;
    });
    const onReply = vi.fn();
    const sender = createSender(send, onReply, vi.fn());

    sender.push({ yaw_deg: 1 });
    sender.push({ yaw_deg: 2 });
    sender.push({ pitch_deg: 5 });
    expect(send).toHaveBeenCalledTimes(1);
    expect(sender.busy).toBe(true);

    replies[0]?.resolve('first');
    await flush();
    expect(calls).toEqual([{ yaw_deg: 1 }, { yaw_deg: 2, pitch_deg: 5 }]);

    replies[1]?.resolve('second');
    await flush();
    expect(onReply).toHaveBeenCalledWith('second');
    expect(sender.busy).toBe(false);
  });

  it('skips a reply that a queued patch has already superseded', async () => {
    const replies: ReturnType<typeof deferred<string>>[] = [];
    const send = vi.fn(() => {
      const d = deferred<string>();
      replies.push(d);
      return d.promise;
    });
    const onReply = vi.fn();
    const sender = createSender(send, onReply, vi.fn());

    sender.push({ a: 1 });
    sender.push({ a: 2 });
    replies[0]?.resolve('A');
    await flush();
    expect(onReply).not.toHaveBeenCalledWith('A');

    replies[1]?.resolve('B');
    await flush();
    expect(onReply).toHaveBeenCalledWith('B');
  });

  it('reports errors and keeps going', async () => {
    const onError = vi.fn();
    const send = vi.fn().mockRejectedValueOnce(new Error('boom')).mockResolvedValueOnce('ok');
    const onReply = vi.fn();
    const sender = createSender(send, onReply, onError);
    sender.push({ a: 1 });
    await flush();
    expect(onError).toHaveBeenCalledWith(new Error('boom'));
    sender.push({ a: 2 });
    await flush();
    expect(onReply).toHaveBeenCalledWith('ok');
  });

  it('drop() discards the queued patch', async () => {
    const first = deferred<string>();
    const send = vi.fn().mockReturnValueOnce(first.promise);
    const sender = createSender(send, vi.fn(), vi.fn());
    sender.push({ a: 1 });
    sender.push({ a: 2 });
    sender.drop();
    first.resolve('done');
    await flush();
    expect(send).toHaveBeenCalledTimes(1);
  });
});
