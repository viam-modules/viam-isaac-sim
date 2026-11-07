import { describe, expect, it } from 'vitest';

import type { View } from './contract';
import { MAX_PITCH_DEG, MIN_DISTANCE, orbit, pan, zoom } from './orbit';

const base: View = { target: [0, 0, 0], distance: 2, yaw_deg: 0, pitch_deg: 0, fov_deg: 70 };

describe('orbit', () => {
  it('maps drag pixels to yaw and pitch', () => {
    const next = orbit(base, 10, -20);
    expect(next.yaw_deg).toBeCloseTo(-3);
    expect(next.pitch_deg).toBeCloseTo(-6);
  });

  it('clamps pitch and wraps yaw', () => {
    expect(orbit(base, 0, 1000).pitch_deg).toBe(MAX_PITCH_DEG);
    expect(orbit({ ...base, yaw_deg: 179 }, -10, 0).yaw_deg).toBeCloseTo(-178);
  });
});

describe('zoom', () => {
  it('scales distance and keeps a minimum', () => {
    expect(zoom(base, 1.1).distance).toBeCloseTo(2.2);
    expect(zoom(base, 0).distance).toBe(MIN_DISTANCE);
  });
});

describe('pan', () => {
  it('moves the target in the screen plane', () => {
    // camera at +X looking at the origin: screen right is +Y, screen up is +Z
    const right = pan(base, 100, 0);
    expect(right.target[0]).toBeCloseTo(0);
    expect(right.target[1]).toBeCloseTo(-0.3);
    expect(right.target[2]).toBeCloseTo(0);

    const up = pan(base, 0, 100);
    expect(up.target[2]).toBeCloseTo(0.3);
  });
});
