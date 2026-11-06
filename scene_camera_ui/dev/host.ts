import { createHost } from '@viamrobotics/post-message-sdk/host';
import { describeRobot, getRobotClient } from './robot';

const resourceName = import.meta.env.VITE_RESOURCE_NAME ?? 'my-sensor';
const iframe = document.querySelector('iframe');
const status = document.querySelector('[role="status"]');
if (!iframe || !status) throw new Error('host.html needs an iframe and a [role="status"].');

try {
  const robot = await getRobotClient(resourceName);
  createHost(robot, iframe, {
    resourceName,
    allow: ({ resource }) => resource === resourceName,
    onSize: (height) => {
      iframe.style.height = `${String(height)}px`;
    },
  });
  status.textContent = describeRobot();
} catch (error) {
  status.textContent = `Could not connect: ${error instanceof Error ? error.message : String(error)}`;
}
