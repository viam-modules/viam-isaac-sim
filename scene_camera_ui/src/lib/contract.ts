/** Replies from the viam-labs:isaac-sim:scene_camera DoCommand contract. */
export type Vec3 = [number, number, number];

export interface View {
  target: Vec3;
  distance: number;
  yaw_deg: number;
  pitch_deg: number;
  fov_deg: number;
}

export interface ViewReply extends View {
  home: View;
  recording: boolean;
}

export type ViewPatch = Partial<View>;

/** The module's capture-guard error text, used to recognise the recording lock. */
export const RECORDING_ERROR = 'during episode capture';

export function toView(reply: ViewReply): View {
  const { target, distance, yaw_deg, pitch_deg, fov_deg } = reply;
  return { target: [...target], distance, yaw_deg, pitch_deg, fov_deg };
}
