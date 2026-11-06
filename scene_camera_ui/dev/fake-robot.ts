import { ResourceName, Struct, type RobotClient } from '@viamrobotics/sdk';

const SENSOR_SERVICE = 'viam.component.sensor.v1.SensorService';
const BASE_TEMPERATURE = 20;
const TEMPERATURE_STEP = 0.5;
const TEMPERATURE_STEPS = 10;
const HUMIDITY = 40;

/** A robot with one sensor, `resourceName`, whose temperature changes on every `getReadings`. */
export function createFakeRobot(resourceName: string): RobotClient {
  let calls = 0;
  const sensorService = {
    getReadings() {
      calls += 1;
      const temperature = BASE_TEMPERATURE + (calls % TEMPERATURE_STEPS) * TEMPERATURE_STEP;
      return { readings: Struct.fromJson({ temperature, humidity: HUMIDITY }).fields };
    },
  };

  // Single cast at the fake boundary: the host only calls these two methods, and a full
  // RobotClient needs a live connection.
  return {
    createServiceClient: (service: { typeName: string }) =>
      service.typeName === SENSOR_SERVICE ? sensorService : undefined,
    resourceNames: () =>
      Promise.resolve([
        new ResourceName({
          namespace: 'rdk',
          type: 'component',
          subtype: 'sensor',
          name: resourceName,
        }),
      ]),
  } as unknown as RobotClient;
}
