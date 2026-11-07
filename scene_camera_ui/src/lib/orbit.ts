import type { Vec3, View } from './contract';

export const ORBIT_DEG_PER_PX = 0.3;
export const KEY_ORBIT_DEG = 5;
export const ZOOM_STEP = 1.1;
/** Target travel per dragged pixel, as a fraction of the camera distance. */
export const PAN_PER_PX = 0.0015;
export const MIN_DISTANCE = 0.05;
export const MAX_PITCH_DEG = 89;

const RAD = Math.PI / 180;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const wrapDeg = (deg: number) => ((((deg + 180) % 360) + 360) % 360) - 180;

export function orbit(view: View, dx: number, dy: number): View {
  return {
    ...view,
    yaw_deg: wrapDeg(view.yaw_deg - dx * ORBIT_DEG_PER_PX),
    pitch_deg: clamp(view.pitch_deg + dy * ORBIT_DEG_PER_PX, -MAX_PITCH_DEG, MAX_PITCH_DEG),
  };
}

export function zoom(view: View, factor: number): View {
  return { ...view, distance: Math.max(MIN_DISTANCE, view.distance * factor) };
}

/** Screen-right and screen-up unit vectors for a camera looking at the target. */
function screenAxes(view: View): { right: Vec3; up: Vec3 } {
  const yaw = view.yaw_deg * RAD;
  const pitch = view.pitch_deg * RAD;
  // forward = target - position
  const f: Vec3 = [
    -Math.cos(pitch) * Math.cos(yaw),
    -Math.cos(pitch) * Math.sin(yaw),
    -Math.sin(pitch),
  ];
  // right = forward x +Z, normalised; up = right x forward
  const rx = f[1];
  const ry = -f[0];
  const rLen = Math.hypot(rx, ry) || 1;
  const right: Vec3 = [rx / rLen, ry / rLen, 0];
  const up: Vec3 = [
    right[1] * f[2] - right[2] * f[1],
    right[2] * f[0] - right[0] * f[2],
    right[0] * f[1] - right[1] * f[0],
  ];
  return { right, up };
}

/** Drag right moves the scene right (the target moves left), like grabbing it. */
export function pan(view: View, dx: number, dy: number): View {
  const { right, up } = screenAxes(view);
  const k = view.distance * PAN_PER_PX;
  const target = view.target.map(
    (t, i) => t - (right[i] ?? 0) * dx * k + (up[i] ?? 0) * dy * k,
  ) as Vec3;
  return { ...view, target };
}
