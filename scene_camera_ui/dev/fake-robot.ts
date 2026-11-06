import { ResourceName, Struct, type JsonValue, type RobotClient } from '@viamrobotics/sdk';

import type { View, ViewReply } from '../src/lib/contract';
import { MAX_PITCH_DEG, MIN_DISTANCE } from '../src/lib/orbit';

const CAMERA_SERVICE = 'viam.component.camera.v1.CameraService';
const WIDTH = 640;
const HEIGHT = 480;
const HOME: View = { target: [0, 0, 0.75], distance: 3, yaw_deg: 37, pitch_deg: 25, fov_deg: 70 };

/**
 * A robot with one scene camera, `resourceName`. Frames are drawn with the current view printed
 * on them. Open the dev host with `?recording` in the URL to see the locked state.
 */
export function createFakeRobot(resourceName: string): RobotClient {
  const recording = new URLSearchParams(location.search).has('recording');
  let view: View = structuredClone(HOME);
  const reply = (): ViewReply => ({ ...view, home: HOME, recording });

  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext('2d')!;

  async function frame(): Promise<Uint8Array> {
    const gradient = ctx.createLinearGradient(0, 0, 0, HEIGHT);
    gradient.addColorStop(0, '#2b3442');
    gradient.addColorStop(1, '#151a21');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    // a horizon that tilts and slides with the view, so orbiting visibly changes the frame
    const horizon = HEIGHT / 2 + view.pitch_deg * 4;
    ctx.strokeStyle = '#8a8f98';
    ctx.beginPath();
    ctx.moveTo(0, horizon);
    ctx.lineTo(WIDTH, horizon);
    ctx.stroke();
    const size = 4000 / (view.distance * view.fov_deg);
    ctx.fillStyle = '#4f78c9';
    ctx.fillRect(WIDTH / 2 - size / 2 + view.yaw_deg, horizon - size, size, size);
    ctx.fillStyle = '#e8eaed';
    ctx.font = '14px monospace';
    ctx.fillText(
      `yaw ${view.yaw_deg.toFixed(1)}  pitch ${view.pitch_deg.toFixed(1)}  d ${view.distance.toFixed(2)}`,
      12,
      HEIGHT - 14,
    );
    const blob = await new Promise<Blob>((resolve) =>
      canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.85),
    );
    return new Uint8Array(await blob.arrayBuffer());
  }

  const cameraService = {
    // CameraClient.getImages reads only `images` and `responseMetadata` off the response
    async getImages() {
      return {
        images: [{ sourceName: resourceName, image: await frame(), mimeType: 'image/jpeg' }],
      };
    },
    // CameraClient.doCommand wraps the command in a real Struct before it reaches the service
    doCommand(request: { command?: Struct }) {
      const cmd = (request.command?.toJson() ?? {}) as Record<string, JsonValue>;
      const { command, ...patch } = cmd;
      if (command !== 'get_view' && recording) {
        throw new Error(`camera '${resourceName}' cannot be reconfigured during episode capture`);
      }
      if (command === 'reset_view') view = structuredClone(HOME);
      if (command === 'set_view') {
        const next = { ...view, ...(patch as Partial<View>) };
        next.distance = Math.max(MIN_DISTANCE, next.distance);
        next.pitch_deg = Math.min(MAX_PITCH_DEG, Math.max(-MAX_PITCH_DEG, next.pitch_deg));
        view = next;
      }
      return { result: Struct.fromJson(reply() as unknown as JsonValue) };
    },
  };

  return {
    createServiceClient: (service: { typeName: string }) =>
      service.typeName === CAMERA_SERVICE ? cameraService : undefined,
    resourceNames: () =>
      Promise.resolve([
        new ResourceName({
          namespace: 'rdk',
          type: 'component',
          subtype: 'camera',
          name: resourceName,
        }),
      ]),
  } as unknown as RobotClient;
}
