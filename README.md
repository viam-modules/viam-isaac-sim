# viam-isaac-sim

A [Viam](https://www.viam.com) module for controlling and simulating robots in
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

What it is/does
====
* Lets you use Viam to control robots in NVIDIA Isaac Sim
* Two pieces
  * `viam_isaac_server`, an Isaac Sim (Kit) extension in [`exts/`](exts/), that
    runs inside your Isaac Sim and serves the module
  * the Viam module: a core model (`world`) that connects to that extension,
    plus a model for each arm, camera, base or other component you want to
    control from Viam
* User does:
  * launches Isaac Sim however they like, with the extension enabled
  * creates the world component in Viam, pointing at that Isaac Sim
  * adds their components, e.g. an arm with `"asset": "ur20"`
  * the component models attach to prims already in the stage, or spawn them
  * controls robots and sees cameras through the normal Viam APIs

## Models

| Model | Viam API | What it does |
|---|---|---|
| `erh:isaac-sim:world` | `generic` | Connects to a running Isaac Sim through the `viam_isaac_server` extension. Configure exactly one. |
| `erh:isaac-sim:arm` | `arm` | Spawns (or attaches to) an articulation - UR arms, Franka, or any USD - and exposes joint control. |
| `erh:isaac-sim:camera` | `camera` | Creates (or attaches to) a camera prim and serves its RGB frames. |
| `erh:isaac-sim:base` | `base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |

Known assets (usable via the `asset` attribute): `ur3e`, `ur5e`, `ur10`,
`ur10e`, `ur16e`, `ur20`, `franka`, `jetbot`. Anything else can be loaded with
`usd_path`, or attach to prims already in your stage with `prim_path`.

## Example machine config

```json
{
  "components": [
    {
      "name": "sim-world",
      "model": "erh:isaac-sim:world",
      "type": "generic",
      "attributes": {
        "address": "localhost:47800"
      }
    },
    {
      "name": "my-ur20",
      "model": "erh:isaac-sim:arm",
      "type": "arm",
      "frame": { "parent": "world" },
      "attributes": {
        "world": "sim-world",
        "asset": "ur20"
      }
    },
    {
      "name": "overhead-cam",
      "model": "erh:isaac-sim:camera",
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
      "model": "erh:isaac-sim:base",
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
| `mock` | `false` | run without Isaac Sim (development/testing) |
| `address` | `localhost:47800` | `host:port` of the `viam_isaac_server` extension |
| `connect_timeout_sec` | `10` | how long to wait for the extension to answer |
| `props` | _none_ | objects to add to the scene (see the pick-and-place fragment) |

How Isaac Sim itself runs (headless, livestream, stage, physics rates) is up
to whoever launches it; the old launch attributes (`headless`, `livestream*`,
`usd_stage`, `physics_dt`, ...) are ignored with a warning.

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

### existing prims

The extension never opens a stage, steps physics or resets the world on its
own - the stage is Isaac Sim's. Each component attaches to the prim at its
`prim_path` (default `/World/<component name>`) if one exists, and only spawns
its `asset`/`usd_path` when it doesn't, so restarts and reconfigures never
duplicate anything. Existing prims are left where they are; the module logs a
warning when that disagrees with the component's frame config. Props likewise
are only added when no prim with their name exists.

Arms and bases need the simulation playing: press Play in Isaac Sim or send
the world `{"command": "play"}`. If Isaac Sim restarts or opens a new stage,
the module reconnects and re-attaches on the next call.

### base attributes

`world` (required), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

## Pick-and-place fragment

The `isaac-sim-pick-and-place` fragment (source in
`fragments/pick-and-place.json`) is a ready-made scene: a UR20 (`pick-arm`)
at the origin, a red 6cm cube to pick up, a flat blue pad to place it on, and
a `scene-cam` watching the workspace. Add the fragment to a machine whose
Isaac Sim is running the extension and everything is spawned on connect. The
stage needs a ground plane (e.g. Create > Physics > Ground Plane) or the cube
falls forever.

Props are configured on the world with the `props` attribute (cubes or USD
references, fixed or dynamic) - see the fragment for the shape of it.

## Running Isaac Sim with the extension

The extension lives in [`exts/viam_isaac_server`](exts/viam_isaac_server)
and needs Isaac Sim 4.5 or newer. Copy this repo's `exts/` directory to the
Isaac Sim machine and add it when launching:

```sh
./isaac-sim.sh --ext-folder /path/to/viam-isaac-sim/exts --enable viam_isaac_server
```

Any Kit launch takes the same flags, e.g. a headless livestreaming one. To
enable it from a standalone Python script instead, after creating the
`SimulationApp`:

```python
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.add_path("/path/to/viam-isaac-sim/exts")
manager.set_extension_enabled_immediate("viam_isaac_server", True)
```

The extension handles requests between app updates, so a standalone script
must keep updating the app (`simulation_app.update()`, or
`world.step(render=True)`).

It listens on `127.0.0.1:47800`. If the module runs on a different machine,
make it listen on a reachable interface:

```sh
--/exts/viam_isaac_server/host=0.0.0.0 --/exts/viam_isaac_server/port=47800
```

There is no authentication, so only do that on a trusted network.

## Viewing the simulator

* **Through Viam (recommended)**: add an `erh:isaac-sim:camera` component with
  `position` + `target` (see the example config) and watch it in the Viam app
  like any other camera - control tab, data capture, SDKs, everything works.
* **Isaac Sim itself**: its own window, or NVIDIA's
  [livestream clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
  if you launched it headless with streaming.

## How it works

* The extension runs inside Isaac Sim. It accepts connections on a
  background thread and handles each request on Kit's main thread, the only
  place Isaac's APIs may be called.
* The module is plain Python with no Isaac dependency. Every component call
  becomes a request to the extension over one TCP connection (the format is
  in `protocol.py`, a file shared by both sides).
* All models in the module share that connection through a singleton, so
  arms/cameras/bases just name their world component and get attached.

## Machine requirements

**The module** runs anywhere viam-server and Python 3.11 do; `first_run.sh`
installs [uv](https://docs.astral.sh/uv/) and the module's dependencies.

**Isaac Sim** you install and run yourself, on the same machine or another
one the module can reach on TCP 47800:

* Ubuntu 22.04/24.04 on x86_64 with an RTX-capable NVIDIA GPU (8GB+ VRAM
  minimum, RTX 4080+/L40 recommended), 32GB+ RAM. See NVIDIA's
  [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
* An NVIDIA driver from a branch Isaac Sim validates against - **use the 580
  branch**. Newer branches (590/595+) are known to crash Isaac's RTX renderer
  on startup (`librtx.scenedb.plugin.so`) and break CUDA init
  (`cuDeviceGetUuid` Warp errors); see
  [isaac-sim/IsaacSim#537](https://github.com/isaac-sim/IsaacSim/issues/537).
  If you're on 595+: `sudo apt-get install -y nvidia-driver-580 && sudo reboot`.

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

- [x] run against an Isaac Sim you launch (`viam_isaac_server` extension)
- [x] play/pause/reset, add_usd, props
- [x] arm joint control (UR family, Franka, arbitrary USD articulations)
- [x] RGB cameras
- [x] differential-drive bases
- [x] cloud builds / registry publishing (tag a release)
- [x] serve kinematics files (`GetKinematics`) so Viam's motion service can do
      IK and planning for simulated arms (all motion stays in Viam, not Isaac)
- [ ] depth / point clouds from cameras
- [ ] gripper support
