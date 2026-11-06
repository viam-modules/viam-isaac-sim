import { createRobotClient, type RobotClient } from '@viamrobotics/sdk';
import { createFakeRobot } from './fake-robot';

const SIGNALING_ADDRESS = 'https://app.viam.com:443';

interface Env {
  host: string;
  apiKey: string;
  apiKeyId: string;
}

/** The `VITE_VIAM_*` credentials, or `undefined` when any of the three is unset. */
const readEnv = (): Env | undefined => {
  const host = import.meta.env.VITE_VIAM_HOST;
  const apiKey = import.meta.env.VITE_VIAM_API_KEY;
  const apiKeyId = import.meta.env.VITE_VIAM_API_KEY_ID;
  return host && apiKey && apiKeyId ? { host, apiKey, apiKeyId } : undefined;
};

/** Dials the machine named by `.env`, or returns a fake robot serving `resourceName`. */
export async function getRobotClient(resourceName: string): Promise<RobotClient> {
  const env = readEnv();
  if (!env) return createFakeRobot(resourceName);
  return createRobotClient({
    host: env.host,
    signalingAddress: SIGNALING_ADDRESS,
    credentials: { type: 'api-key', authEntity: env.apiKeyId, payload: env.apiKey },
  });
}

/** The status line for the robot `getRobotClient` returns. */
export const describeRobot = (): string => {
  const env = readEnv();
  return env ? `Connected to ${env.host}` : 'Connected to a fake robot (set VITE_VIAM_* in .env)';
};
