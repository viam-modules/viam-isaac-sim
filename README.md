# viam-isaac-sim

A [Viam](https://www.viam.com) module for controlling and simulating robots in
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

What it is/does
====
* Lets you use Viam to control robots in NVIDIA Isaac Sim
* Models
  * Core simulator model (`world`) that configures the world and runs Isaac Sim
  * A model for each arm, camera, base or other component you want to control/simulate from Viam
* User does:
  * creates the core sim/world component in Viam
  * adds their components, e.g. an arm with `"asset": "ur20"`
  * the world model starts the sim and the component models spawn the right prims
  * views the simulator via the built-in WebRTC livestream or through Viam camera components
  * controls robots and sees cameras through the normal Viam APIs

## Models

| Model | Viam API | What it does |
|---|---|---|
| `viam-labs:isaac-sim:world` | `generic` | Boots Isaac Sim, opens the USD stage, runs the sim loop. Configure exactly one. |
| `viam-labs:isaac-sim:arm` | `arm` | Spawns (or attaches to) an articulation - UR arms, Franka, or any USD - and exposes joint control. |
| `viam-labs:isaac-sim:camera` | `camera` | Creates (or attaches to) a camera prim and serves its RGB frames. |
| `viam-labs:isaac-sim:base` | `base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |

Known assets (usable via the `asset` attribute): `ur3e`, `ur5e`, `ur10`,
`ur10e`, `ur16e`, `ur20`, `franka`, `jetbot`. Anything else can be loaded with
`usd_path`, or attach to prims already in your stage with `prim_path`.

## Example machine config

```json
{
  "components": [
    {
      "name": "sim-world",
      "model": "viam-labs:isaac-sim:world",
      "type": "generic",
      "attributes": {
        "headless": true,
        "livestream": true
      }
    },
    {
      "name": "my-ur20",
      "model": "viam-labs:isaac-sim:arm",
      "type": "arm",
      "frame": { "parent": "world" },
      "attributes": {
        "world": "sim-world",
        "asset": "ur20"
      }
    },
    {
      "name": "overhead-cam",
      "model": "viam-labs:isaac-sim:camera",
      "type": "camera",
      "frame": {
        "parent": "world",
        "translation": { "x": 2000, "y": 2000, "z": 2000 }
      },
      "attributes": {
        "world": "sim-world",
        "target": [0, 0, 0.5],
        "width": 1280,
        "height": 720
      }
    },
    {
      "name": "my-jetbot",
      "model": "viam-labs:isaac-sim:base",
      "type": "base",
      "frame": {
        "parent": "world",
        "translation": { "x": 1000, "y": 0, "z": 100 }
      },
      "attributes": {
        "world": "sim-world",
        "asset": "jetbot"
      }
    }
  ]
}
```

Every non-world component must set `"world"` to the world component's name.
That attribute is also returned as an implicit dependency from each model's
validate, so viam-server starts the world first - no `depends_on` needed.

Components are **placed with the standard frame config** (translations in mm,
any orientation representation) - the spawn pose in Isaac and viam's frame
system then agree, so things like the motion service see components where
they actually are. The `position` (meters) / `orientation_rpy_deg` attributes
still work as a fallback when no frame is set; a camera `target` attribute
overrides orientation to aim at a point.

### world attributes

| attribute | default | notes |
|---|---|---|
| `mock` | `false` | run without Isaac Sim installed (development/testing) |
| `headless` | `true` | no local GUI window |
| `livestream` | `true` | serve the Isaac Sim WebRTC Streaming Client (headless only) |
| `livestream_public_ip` | _auto-detected_ | IP advertised to streaming clients |
| `livestream_width` / `livestream_height` | `1280` / `720` | streamed resolution |
| `usd_stage` | _empty stage + ground plane_ | USD file or omniverse:// URL to open |
| `physics_dt` / `rendering_dt` | `1/60` | step sizes in seconds |
| `boot_timeout_sec` | `300` | Isaac Sim can take a while on first boot |
| `kit_log_level` | `warning` | kit console verbosity |

The world also supports `DoCommand`: `{"command": "status" | "play" | "pause" |
"reset"}` and `{"command": "add_usd", "usd_path": "...", "prim_path":
"/World/thing", "position": [x, y, z]}` to drop extra props into the scene.

### arm attributes

`world` (required), one of `asset` / `usd_path` / `prim_path`, plus optional
`position` ([x,y,z] meters), `end_effector_prim` (prim path whose world pose is
reported by `GetEndPosition`, converted to Viam's orientation-vector
convention), and `move_timeout_sec`.

`MoveToJointPositions` / `GetJointPositions` work today. IK and motion
planning are deliberately left to Viam (the motion service), not Isaac - the
module's job is just to expose the simulated arm.

`GetKinematics` works: for `ur3e`/`ur5e`/`ur20` the official viam SVA
kinematics files are fetched automatically (and cached in the module data
dir); for anything else set `kinematics_url` to an SVA `.json` or `.urdf`
(http(s):// or file://). With kinematics served, the motion service can plan
for the simulated arm. Module-level `MoveToPosition` still raises - use the
motion service.

### camera attributes

`world` (required), and either `prim_path` of an existing camera in your stage
or `position` plus `target` (aim-at point) or `orientation_rpy_deg` to create
one. `width`/`height` default to 640x480.

### base attributes

`world` (required), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

## Pick-and-place fragment

The `isaac-sim-pick-and-place` fragment (source in
`fragments/pick-and-place.json`) is a ready-made scene: a UR20 (`pick-arm`)
at the origin, a red 6cm cube to pick up, a flat blue pad to place it on, and
a `scene-cam` watching the workspace. Add the fragment to any machine that
meets the requirements above and the world spawns everything at boot.

Props are configured on the world with the `props` attribute (cubes or USD
references, fixed or dynamic) - see the fragment for the shape of it.

## Viewing the simulator

* **Through Viam (recommended)**: add an `viam-labs:isaac-sim:camera` component with
  `position` + `target` (see the example config) and watch it in the Viam app
  like any other camera - control tab, data capture, SDKs, everything works.
* **Full interactive viewport**: install NVIDIA's
  [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
  and connect it to the sim machine's IP (plain IP, no port - TCP 49100 and
  UDP 47998 are hardcoded and must be reachable). If the machine has multiple
  interfaces, set `livestream_public_ip` on the world.
* **Local GUI**: set `"headless": false` on the world (needs a display on the
  sim machine).

## How it works

Isaac Sim's Python API only runs inside Isaac Sim's own interpreter, and
Omniverse Kit wants to own the thread it runs on. So:

* `run.sh` launches the module with Isaac Sim's bundled python (found via
  `$ISAAC_SIM_PATH/python.sh` or `$ISAAC_PYTHON`), installing `viam-sdk` into
  it on first run.
* The **main thread** runs the simulation loop (`SimulationApp` boot, stepping,
  and a task queue). The **Viam module server** runs on a side thread; all
  component calls are marshalled onto the sim thread.
* All models live in one module process and share the sim through a singleton,
  so arms/cameras/bases just name their world component and get attached.

## Machine requirements & automatic setup

On a standard Ubuntu 22.04/24.04 x86_64 machine, the module sets itself up:
when first installed, viam-server runs `first_run.sh`, which installs the
system libraries kit needs (vulkan/GL), the right python (via deadsnakes on
24.04), an NVIDIA driver if none is present (the validated 580 branch - newer
is not better here, see below), and Isaac Sim itself
(pip-installed into a venv under the module's data directory - 4.5.0 on
22.04, 5.0.0 on 24.04). `run.sh` finds that install automatically; the EULA
is accepted via environment variable.

Notes on the automatic setup:

* The Isaac Sim download is 10GB+. If it exceeds viam-server's default
  first-run timeout, set `"first_run_timeout": "2h0m0s"` on the module entry
  in your machine config.
* If the script had to install the NVIDIA driver, **reboot** before
  configuring the world component.
* Already have Isaac Sim? Set `ISAAC_SIM_PATH` (dir containing `python.sh`)
  or `ISAAC_PYTHON` in the module's environment variables and the script
  skips everything.

What the machine must already be/have (the script can't do these for you):

* Ubuntu 22.04 or 24.04 on x86_64 with an RTX-capable NVIDIA GPU (8GB+ VRAM
  minimum, RTX 4080+/L40 recommended), 32GB+ RAM, ~60GB free disk. See
  NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
* An NVIDIA driver from a branch Isaac Sim validates against - **use the 580
  branch**. Newer branches (590/595+) are known to crash Isaac's RTX renderer
  on startup (`librtx.scenedb.plugin.so`) and break CUDA init
  (`cuDeviceGetUuid` Warp errors); see
  [isaac-sim/IsaacSim#537](https://github.com/isaac-sim/IsaacSim/issues/537).
  If you're on 595+: `sudo apt-get install -y nvidia-driver-580 && sudo reboot`.
* viam-server installed, machine online in the Viam app, running as root (or
  a user with passwordless sudo) for the apt/driver steps.
* Network access to pypi.nvidia.com, pypi.org, and NVIDIA's asset servers.
* Open ports for the livestream viewer if you want it: TCP 49100 (signaling)
  plus UDP 47998 (media) - both hardcoded in NVIDIA's streaming client.

No GPU/Isaac at all? `"mock": true` on the world runs the module anywhere for
development.

## Development without Isaac Sim (mock mode)

Set `"mock": true` on the world component and the module runs anywhere python
does - arms integrate joint targets over time, cameras produce synthetic
frames, bases accept velocity commands. This is what the test suite uses:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests/
```

## Status / roadmap

- [x] world boot, stage loading, livestream, play/pause/reset, add_usd
- [x] arm joint control (UR family, Franka, arbitrary USD articulations)
- [x] RGB cameras
- [x] differential-drive bases
- [x] cloud builds / registry publishing (tag a release)
- [x] serve kinematics files (`GetKinematics`) so Viam's motion service can do
      IK and planning for simulated arms (all motion stays in Viam, not Isaac)
- [ ] depth / point clouds from cameras
- [ ] gripper support
