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
| `viam-labs:isaac-sim:gripper` | `gripper` | Exposes a supported parallel gripper, including the Lite 6's integrated gripper. |
| `viam-labs:isaac-sim:camera` | `camera` | Creates (or attaches to) a camera prim and serves its RGB frames. |
| `viam-labs:isaac-sim:base` | `base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |

Known robot assets (usable via the `asset` attribute): `lite6`, `ur3e`, `ur5e`,
`ur10`, `ur10e`, `ur16e`, `ur20`, `franka`, and `jetbot`. Known grippers
are `lite6_gripper` and `robotiq_2f_85`. The Lite 6 assets require Isaac Sim
5.0 or newer. Anything else can be loaded with `usd_path`, or attached to prims
already in your stage with `prim_path`.

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

### Sim/real config toggler

`config_toggler.py` transforms local JSON or a Viam Cloud machine-part config.
The initial `lite6-sim-real-mapping.json` maps the Lite 6 arm, fixed camera,
and wrist camera from `lite-realsense-wrist-mount.json` to their simulator
equivalents. The simulator world, policy, and modules are not mapped yet.
Local fragment sources can be supplied before transformation with `--fragment
FRAGMENT_ID:PATH ...`; `--part-id` instead downloads the part config and all
referenced fragments. The transformer applies fragment variables and
`$set`/`$unset` modifications in memory, but preserves the cloud fragment
references in its output. Generated local twins stay top-level; toggler-owned
fragment modifications enable or disable fragment-backed real resources.
Mapping entries with `"copy_real_frame_to_sim": true` copy the effective real
component frame—including geometry—onto the generated sim twin before
reference rewriting. This keeps measured placement authoritative while the sim
template remains the fallback when no real component is present.
For a simulated camera with `parent_prim`, a copied non-world frame is applied
as the camera's local pose in Viam optical axes (X right, Y down, Z forward).
`parent_prim` selects the moving USD parent; it no longer requires a duplicate
`local_position` or `local_orientation_rpy_deg`.

When an `viam-labs:isaac-sim:world` component named `sim-world` is present, enabled
world-parented `rdk:builtin:fake` generic components with axis-aligned box
geometry are also emitted into its existing `attributes.props` list as fixed
Isaac cubes. Generated entries have `"from_toggler": true`; each run replaces
those entries while preserving authored props. Dimensions and positions are
converted from Viam millimeters to Isaac meters. Rotated boxes are rejected
until orientation composition is supported.
The mapping's `static_materials` object optionally assigns the lightweight
`wood` and `brushed_metal` USD Preview Surface presets to generated props. These
presets set base color, roughness, and metallic response; they do not use image
textures.

```sh
# Add/update twins while preserving the currently active lane.
./config_toggler.py sync --mapping lite6-sim-real-mapping.json input.json -o output.json

# Read one or more cloud fragments locally while preserving their references.
./config_toggler.py sync --mapping lite6-sim-real-mapping.json live-config.json \
  --fragment 1141b2a7-1b20-4047-8766-12da9524c80f:lite-realsense-wrist-mount.json \
  -o synced-config.json

# Preview a cloud transformation using credentials from a viam.json file.
./config_toggler.py set --to sim --mapping lite6-sim-real-mapping.json \
  --part-id PART_ID --creds viam-twinning-demo.json --dry-run

# Update the part only if its cloud config has not changed since download.
./config_toggler.py set --to sim --mapping lite6-sim-real-mapping.json \
  --part-id PART_ID --creds viam-twinning-demo.json --write-cloud

# Update cloud and also save the resulting JSON locally.
./config_toggler.py sync --mapping lite6-sim-real-mapping.json \
  --part-id PART_ID --creds viam-twinning-demo.json --write-cloud \
  -o synced-config.json

# Select a lane repeatedly without flip-state ambiguity.
./config_toggler.py set --to sim --mapping lite6-sim-real-mapping.json input.json -o output.json
./config_toggler.py set --to real --mapping lite6-sim-real-mapping.json input.json -o output.json

# Exit 1 when sync or the requested lane would change the input; exit 2 on invalid input.
./config_toggler.py check --to sim --mapping lite6-sim-real-mapping.json input.json
```

Use `-` as the local input for stdin. `--creds` reads
`cloud.api_key.key` and `cloud.api_key.id` from a viam.json file and sets
`VIAM_API_KEY` and `VIAM_API_KEY_ID`; without it, the existing environment is
used. Cloud mode is read-only unless `--write-cloud` is present. `--dry-run`
always prints and suppresses local and cloud writes. A cloud write is skipped
when the transformation makes no change and uses the downloaded part's update
timestamp to reject concurrent edits. With `--write-cloud`, JSON is emitted
only when `-o` or `--dry-run` is also requested. Output-file replacement is
atomic. Only reference fields declared by the mapping are rewritten; an
undeclared field containing a mapped resource name is rejected rather than
changed recursively.

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
| `livestream` | `true` | WebRTC viewer at `http://<host>:8211/streaming/webrtc-client` (headless only) |
| `livestream_public_ip` | _auto-detected_ | IP advertised to streaming clients |
| `livestream_width` / `livestream_height` | `1280` / `720` | streamed resolution |
| `dome_light` | `false` | add a neutral dome light at `/World/DomeLight` with intensity 1000 |
| `usd_stage` | _empty stage + ground plane_ | USD file or omniverse:// URL to open |
| `physics_dt` | `1/60` | physics step size in seconds |
| `rendering_dt` | one render per capture sample | render period in seconds; must be an integer multiple of `physics_dt`. Defaults to `1/capture_rate_hz` when that is a whole number of physics steps (`1/20` for 20 Hz capture at 60 Hz physics), else `physics_dt`. Set it explicitly to `physics_dt` for a smooth livestream |
| `capture_rate_hz` | `15` | episode capture rate in simulator time |
| `boot_timeout_sec` | `300` | Isaac Sim can take a while on first boot |
| `kit_log_level` | `warning` | kit console verbosity |

The world supports low-level `DoCommand` operations for pre-created props. Use
`set_prop_pose` to place a prop at a specific world pose:

```json
{
  "command": "set_prop_pose",
  "prop": "pick_block",
  "position": [0.3, 0.0, 0.025],
  "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

`get_prop_pose` returns the current measured world pose, while `park_props`
restores configured spawn poses. Pose changes zero linear and angular velocity.
Task-specific placement and randomization belong in caller logic; see
`examples/scripted_pick.py` for seeded workspace sampling. Existing `status`,
`play`, `pause`, `reset`, and `add_usd` commands remain available.


### arm attributes

`world` (required), one of `asset` / `usd_path` / `prim_path`, plus optional
`position` ([x,y,z] meters), `end_effector_prim` (prim path whose world pose is
reported by `GetEndPosition`, converted to Viam's orientation-vector
convention), `move_timeout_sec`, and `arm_model` as a manifest label for custom
USDs. `max_joint_deg_s` sets one positive finite velocity limit in degrees per
second on every configured arm joint in the Isaac articulation; gripper joints
on an integrated articulation are unchanged. `max_joint_deg_s2` sets the
positive finite per-joint acceleration limit reported to trajectory
parameterizers. `is_moving_velocity_threshold_rad_s` sets the per-joint velocity
noise floor used by `IsMoving`; it defaults to `0.01` rad/s and must be finite
and non-negative. Known UR assets default to `wrist_3_link`; `lite6` selects
NVIDIA's integrated gripper variant and defaults to its gripper body.

`MoveToJointPositions`, `MoveThroughJointPositions`, and
`MoveThroughJointPositionsStreamed` work today. Streamed trajectories use their
timestamps as simulator time: a physics callback applies each complete
`ArticulationAction`, holds the measured joint positions, and clears measured
joint velocity when the trajectory finishes or the request is cancelled. IK
and motion planning are deliberately left to Viam (the motion service), not
Isaac—the module's job is just to expose the simulated arm.

`GetKinematics` works: `lite6`, `ur3e`, `ur5e`, and `ur20` fetch their official
Viam kinematics files automatically and cache them in the module data directory.
For anything else set `kinematics_url` to an SVA `.json` or `.urdf` (http(s)://
or file://). With kinematics served, the motion service can plan
for the simulated arm. Module-level `MoveToPosition` still raises - use the
motion service.

For timestamped controllers, arm `DoCommand` accepts
`{"command":"set_joint_targets_radians","values":[q0,...,q5]}`. It validates
six finite radians targets, latches them, and returns immediately without waiting
for waypoint settlement. `get_joint_positions_radians` returns the matching
radian state used by policy clients.

### gripper attributes

`world` and `arm` are required. `lite6_gripper` exposes the gripper already
composed into NVIDIA's `lite6` asset; `robotiq_2f_85` is mounted under a UR
wrist and accepts an optional local mount pose.

`sensorless` is an optional boolean, defaulting to `false`. When `false`,
`IsHolding` retains the simulator's contact/stall detection. When `true`,
`IsHolding` reports whether a closed grip command is active; `Open`, `Stop`, and
non-closed input commands clear that state. Set `sensorless: true` for the Lite6
to match the production gripper. The module logs a warning when a
`lite6_gripper` is configured without it.

Lite6 policy controls are binary. `Open`, `Grab`, binary `GoToInputs`, and
binary `set_input` latch an open/closed target and return immediately, matching
the production driver's command timing. `GetCurrentInputs` reports the latched
binary state, not measured jaw position. Position-capable grippers retain
continuous normalized `GetCurrentInputs`, `GoToInputs`, and `set_input`.

Internally, Isaac still simulates continuous jaw position and contact. After
bilateral fingertip contact, the Lite6 gripper switches its drive to a continuous
2 N closing effort applied every physics step. The next jaw command clears that
effort and restores the configured position gains.

The Lite6 gripper accepts `jaw_layout`: `narrow`, `middle_positive_y`,
`middle_negative_y`, or `wide` (the default, matching NVIDIA's asset). The two
middle layouts are intentionally named in the gripper's local Y axis for this
simulator spike; the physical module can replace that convention with
plug-relative names once the deployed mounting orientation is measured. The
layout rotates each visual bracket and its compound contact colliders together;
the conservative planning envelope remains unchanged. `jaw_layout` does not
automatically update the configured TCP frame. For this Lite6 simulator model,
planner Y is opposite the USD gripper-body Y. Set `frame.translation` X to `0`
and Z to `83.6` mm, and select Y explicitly with the layout:

| `jaw_layout` | Planner TCP Y (mm) |
| --- | ---: |
| `narrow` | 0 |
| `wide` | 0 |
| `middle_positive_y` | -5.6 |
| `middle_negative_y` | +5.6 |

`lite6-sim-real-mapping.json` selects the symmetric `wide` layout with zero
planner TCP Y offset. The real-robot frame remains unchanged.

For direct `DoCommand` callers, `set_input` accepts only `0` or `1` on Lite6;
position-capable grippers accept any finite value in `[0,1]`. Both forms latch
the target without waiting for settlement.

### camera attributes

`world` is required. Prefer the component's standard `frame` for placement;
without a frame, legacy `position` plus `target` (aim-at point) or
`orientation_rpy_deg` can create and orient a camera. Conflicting frame and
legacy pose fields are rejected. `width`/`height` default to 640x480 and
`near_clip_m` defaults to 0.01 m so close-range robot views are not clipped. A
wrist-mounted camera uses `parent_prim` plus a non-world standard frame. The pick
examples place it outside the tool envelope while centering it on gripper Y.

Frame pose, `target`, `fov_deg`, and `near_clip_m` updates apply live when no
episode is being captured. Changing `prim_path`, `parent_prim`, `width`, or
`height` requires restarting the module. Camera changes are rejected during
native or CosmosWriter episode capture.

### Policy execution

Policy execution is provided by the separate
`awinter:arm-policy:policy` generic component. It depends on this module's
world, arm, gripper, and camera resources but runs in its own lightweight module
process; the Cosmos/OpenPI inference server remains external. See
`../arm-policy/awinter_arm-policy_policy.md` for its configuration and command
contract. `lite6-config.json` shows the two modules wired together.

### Lite6 policy dynamics calibration

Calibrate the Lite6 simulator-only dynamic profile against the loaded Isaac
articulation before changing its versioned limits:

```sh
python3 examples/measure_lite6_dynamics.py <machine-address> \
  --output runs/live/lite6-dynamics-measurement.json
```

The measurement records 20 Hz positions, achieved velocity and acceleration,
tracking error, overshoot, stop displacement, and the articulation's available
drive/controller fields across three safe poses. Its separate physical reference
is the UFACTORY Lite6 Hardware Manual V2.6.0 (joint motion: 180 deg/s and
1145 deg/s²); physical values never authorize policy execution.

`lite6-integrated-gripper-sim-v2` comes from the 2026-09-17 Isaac 5.0 run of
that command. The loaded articulation reported per-joint drive stiffness,
damping, and effort but no explicit velocity or acceleration caps. Across all
three transitions every joint achieved at least 0.593 rad/s and 8.242 rad/s²;
the simulator-only profile therefore uses conservative uniform limits of
0.5 rad/s and 8.0 rad/s². The archived 30m and 60m raw chunks peak at
7.502/9.642 rad/s and 143.247/157.329 rad/s² respectively, so both old
rejections were genuine rather than provisional-limit false positives. The
time parameterizer preserves all 32 waypoints in 152 and 146 samples (7.6 s
and 7.3 s) at 20 Hz; the unchanged workspace gate still rejects any waypoint
whose FK falls below the configured floor.

### base attributes

`world` (required), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

## Pick-and-place fragment

The `isaac-sim-pick-and-place` fragment (source in
`fragments/pick-and-place.json`) is a ready-made scene: a UFactory
Lite 6 (`pick-arm`) with symmetric wide jaws (`pick-gripper`), a red 31.3 mm
cube (`pick_block`), a flat blue pad, and wrist cameras. The cube width is
halfway between the wide layout's 22.4 mm closed gap and 40.2 mm open gap. The
wrist camera and planner TCP are centered on the gripper Y axis.
`park_props` restores the cube to its configured parking pose.
`examples/scripted_pick.py` deterministically samples a reachable pose and
applies it with `set_prop_pose`.

Props are configured on the world with the `props` attribute (cubes or USD
references, fixed or dynamic). Cube dimensions are `size * scale`; optional
`static_friction` and `dynamic_friction` default to `1.0`. See the fragment for
the configuration shape.

Run one deterministic top-down pick through the configured machine's normal
world, gripper, and built-in motion APIs. CosmosWriter output is an explicit
boolean opt-in and requires native episode capture:

```sh
VIAM_API_KEY=... VIAM_API_KEY_ID=... \
  python3 examples/scripted_pick.py <machine-address> --seed 42 \
  --episode-output-dir runs/demo --cosmos-output
```

The included `pick_block` is the default target. Run three seeds serially while
watching the scene camera:

```sh
export VIAM_API_KEY=...
export VIAM_API_KEY_ID=...
for seed in 54 55 56; do
  python3 examples/scripted_pick.py localhost:8080 --disable-webrtc \
    --seed "$seed"
done
```

The action resets `pick_block`, randomizes the wrist's starting point, plans the
centering approach, and opens and settles the gripper before episode capture
starts. All random values use independent streams derived from the episode seed,
so an episode is reproducible and adding one noise source does not perturb the
others.

| Action point | Default noise |
|---|---|
| Start position | Gaussian XYZ; standard deviation 10% of waypoint Z |
| Centering midpoint | Gaussian XYZ; standard deviation 10% of waypoint Z |
| Centering arrival | Gaussian XYZ; standard deviation 5% of waypoint Z |
| Descent midpoint | Gaussian XY; standard deviation 5% of waypoint Z; exact Z |
| Correction-plane arrival | Gaussian XY with 1 cm standard deviation; exact Z; Gaussian yaw with 15° standard deviation |
| Grasp height | Seeded uniform sample from 1.252–1.565 cm |
| Lift destination | Gaussian XYZ; standard deviation 10% of waypoint Z |

The noisy arrival, yaw correction, and position correction all occur at the
pre-grasp clearance plane, 5 cm above grasp height. After correction, one exact
vertical move descends between the open jaws to the grasp pose. Pass
`--no-noise` to disable every row above and use `PickConfig.grasp_height_m`
exactly. The first captured stage begins with centering motion. Every
non-grip-only arm subtask turns its `DoPlan` joint path into a
zero-boundary-velocity TOPP-RA trajectory using the arm's reported velocity and
acceleration limits, samples targets at 20 Hz, and sends one complete batch
through `MoveThroughJointPositionsStreamed`. For compatibility with older arm
modules that omit acceleration metadata, the script uses a conservative
60 degrees per second squared fallback.

Centering and the two-waypoint approach to the correction plane each total at
least 2.2 seconds (33 frames at 15 Hz). Yaw and position corrections each run
for at least 0.183 seconds. The final 5 cm vertical descent runs for at least
0.733 seconds. The grasp subtask closes until the simulator reports a firm
grasp, waits another 0.25 simulator seconds, then streams either the successful
lift or the closed-jaw miss retreat. `--gripper-settle-sec` is the maximum
simulator-time close wait; a timeout is treated as a missed grasp. The wait
also ends as soon as the jaw has moved and stopped without latching (a
closed-jaw miss), and the open wait ends as soon as the jaw stops; a gripper
that does not implement `IsMoving`, or whose `IsMoving` never reports motion,
keeps the full fixed wait, so real-robot runs behave as before. Gripper time
does not count toward the arm-motion duration guarantee. A successful lift
starts the labeled `hold still` stage, which captures a full 33-frame action
chunk before the episode ends.
Simulator physics pauses while plans are computed, so planning latency does not
produce idle capture frames. The included pick fragment caps the arm at 30
degrees per second and 60 degrees per second squared; changing
`max_joint_deg_s` or `max_joint_deg_s2` requires restarting the module.

The firm-grasp signal is reported in `IsHoldingSomething` metadata independently
of sensorless binary `IsHolding` semantics. It exits nonzero when the verified
outcome is failure.
Use `--yaw-offset-deg` if the asset's jaw axis is rotated relative to the block's
long axis.
The pick region is defined relative to the arm base and transformed through
`<arm>:base` into world coordinates once per machine connection. Moving or
rotating the configured arm frame therefore moves the generated target region
without script constants tied to a particular table layout.

Set the simulator world attribute `capture_rate_hz` to control synchronized
episode capture; it defaults to 15 Hz and must be positive, finite, and exactly
representable by the configured `physics_dt`. By default the simulator renders
once per capture sample (`rendering_dt = 1/capture_rate_hz`) rather than every
physics step, since frames between samples are never read.

`batch_picks.py` records per-attempt `started_at`, `wall_time_sec`, and
`home_move_sec` in `run.json`, and each attempt's `result.timing` breaks the
pick down by phase (`phases_sec`), per streamed move (`moves`, with the planned
trajectory duration beside the wall time it took), gripper waits, and simulator
seconds elapsed, so throughput changes can be measured directly:

```sh
python3 -c 'import json,statistics; a=json.load(open("runs/demo/run.json"))["attempts"]; w=[x["wall_time_sec"] for x in a if "wall_time_sec" in x]; print(len(w), "episodes, mean wall", round(statistics.mean(w),2), "s")'
```

With `--episode-output-dir`, the action captures joint position, velocity and
command, end-effector pose, gripper command and observation state, and the
fixed and wrist RGB cameras at `capture_rate_hz` of simulator time. Set
`--fixed-camera-2 third-cam` to capture an optional second fixed view; its
default is empty, so existing two-camera runs are unchanged. Episode schema v3
records the episode-level `task` and contiguous, end-exclusive frame ranges for
three scripted stages: center above the block, rotate-and-descend to the
pre-grasp pose, and final descent/grip/lift. The final descent uses one linear
Cartesian move with the pre-grasp and grasp poses at the same yaw, reducing the
risk of a rotating jaw contacting the block. Every annotated stage contains at
least 33 frames, enough to yield a 32-action training window. Set
the task with `--task`; it defaults to `Pick up the block`. Lite6 gripper
command/state remain binary `0=open`, `1=closed`; continuous jaw physics is not
policy state.
Encoding and Parquet writes use a bounded worker queue. Each episode is written
under `.<episode-id>.partial` and renamed to `<episode-id>` only after
`samples.parquet`, every configured camera MP4 (`fixed.mp4`, `wrist.mp4`, and
optional `fixed_2.mp4`), and `manifest.json` close successfully; a dropped or
inconsistent sample fails the episode and leaves the partial directory unpublished.

`scripted_pick.py` treats `--episode-output-dir` as a run root: native output
goes under `runs/demo/episodes/<episode-id>` and, with `--cosmos-output`, its
CosmosWriter output goes under `runs/demo/cosmoswriter/<episode-id>`. It does
not create `run.json`; use `batch_picks.py` for the complete resumable
dataset-run path.

`--cosmos-output` attaches one multi-render-product CosmosWriter pipeline to
dedicated render products for `--fixed-camera`, `--wrist-camera`, and the
optional `--fixed-camera-2`. It requires `--episode-output-dir` and writes the
matching sibling tree to `<run-root>/cosmoswriter/<episode-id>`; CosmosWriter
files are never written into the native episode directory. Native and
CosmosWriter finalization remain independent. Clips are written under `fixed/`,
`wrist/`, and, when configured, `fixed_2/`, each with a `clip_0000/` directory.
The contiguous `{rgb,depth,edges,segmentation,shaded_seg}_NNNN.png` sequences
are authoritative. After CosmosWriter drains and closes, every modality is
deterministically encoded from those PNGs as H.264/yuv420p at the episode
capture rate; the PNGs remain in place. Successful finalization returns every configured role
in `cosmos_paths`. Mock mode rejects the option.

The manifest derives the arm model, canonical joint names, gripper model, and
gripper mode from the instantiated component assets; callers do not supply
embodiment labels.

Custom capture clients pass `task` to `start_episode`, then await
`mark_episode_stage` with the next stage task at each boundary. The first mark
starts at frame 0. A later mark keeps capturing the completed state until the
current stage reaches 33 frames, then closes that range and starts the next
stage; the simulator thread remains unblocked while the command waits.
`stop_episode` similarly pads the final stage before closing it at
`sample_count`.
Native camera encoding forces a closed-GOP H.264 IDR at every recorded stage
start. CosmosWriter finalization applies the same stage boundaries to every
role and modality, allowing downstream stage clips to be remuxed without
decoding or re-encoding.

Generate a numbered, resumable run through one machine connection:

```sh
python3 examples/batch_picks.py localhost:8080 runs/picks-001 \
  --disable-webrtc --count 100 --start-seed 1000 --noise-fraction 0.6 \
  --upload-dataset-id <dataset-id> --upload-part-id <part-id>
```

`batch_picks.py` derives CosmosWriter output from its positional run directory:

```text
runs/picks-001/
├── run.json
├── episodes/
│   └── 000000/
│       ├── manifest.json
│       ├── samples.parquet
│       ├── fixed.mp4
│       └── wrist.mp4
└── cosmoswriter/
    └── 000000/
        ├── fixed/clip_0000/{rgb,depth,edges,segmentation,shaded_seg}.mp4
        └── wrist/clip_0000/{rgb,depth,edges,segmentation,shaded_seg}.mp4
```

`run.json` records every attempt and `episodes/000000`, `000001`, ... contain
finalized native episodes. New batch runs use schema 5. `--noise-fraction`
accepts a value from 0 through 1 and defaults to `0.6`; each episode's seed
deterministically selects whether that episode receives action noise, so resume
and count expansion preserve all previous selections. Every attempt records its
resolved `noise` boolean. The persisted configuration stores `cosmos_output` and
`noise_fraction`, never a caller-selected CosmosWriter root. With
`cosmos_output: true`, matching CosmosWriter episodes are written to
`cosmoswriter/000000`, `cosmoswriter/000001`, ..., preserving the fixed and
wrist `clip_0000` layouts above. A resumed command must use the same schema-5
configuration, including `--noise-fraction`, except that `--count` may increase
to append more episodes. Decreasing `--count` or changing another setting is
rejected.

`--upload-dataset-id` has the machine upload each finalized native episode to
that Viam dataset as `episode-<episode-id>.zip`. The upload runs inside the
module (`stop_episode`), next to the episode files, using the API key and
machine part ID viam-server provides to the module, so the computer running
`batch_picks.py` needs no extra credentials or filesystem access. Uploads use
component type `generic` and the `--world` component name, are tagged with the
run directory name, `isaac-episode`, and any `--upload-tag` values, and are
retried a few times before being reported as failed. Use `--app-url` if the
machine belongs to a non-production app host.

Each attempt in `run.json` records its upload `file_id` and size, or the upload
error. Resuming asks the module to retry failed uploads (`upload_episode`) and
skips episodes already uploaded to the same dataset; resuming with upload flags
on an existing run uploads its already-finalized episodes. The command exits
nonzero if any upload failed.

When episode files are written on the machine and are not visible from the
computer running `batch_picks.py`, the run records each episode from the
module's `stop_episode` result (marked `on_machine` in `run.json`), and a resume
skips those episodes instead of recapturing them.

For a CosmosWriter-enabled run, the shared finalizer fully decodes and validates
all five modality MP4s for every configured camera before accepting the episode.
Each must have one H.264/yuv420p video stream, the expected frame count, constant
geometry, timing at the manifest `sample_rate_hz`, decoded timestamps spaced by
its reciprocal, and a duration consistent with `frame_count / sample_rate_hz`. All five modalities for a camera
must agree on count and geometry, and all cameras must match the scheduled
Cosmos frame count. That count must also equal the required native sample count.
JSON-native stream facts are recorded in attempt metadata. If encoding or
validation fails, no `cosmos_paths` are returned and
the output, including diagnostic artifacts, is preserved in place.

Re-running the same command skips valid finalized episodes. If a finalized
native episode has missing or invalid required CosmosWriter output, batch records
an actionable error without recapturing or overwriting either output. It
preserves failures and never treats `.partial` directories as finalized. Bulk
generation uses a single serial Viam connection. The pick fragment disables the
Isaac WebRTC livestream by default to avoid its GPU and encoding overhead; set
`livestream` to `true` only for interactive auditing.

The same completed source run can be packaged as either direct LeRobot v3
variant used by Cosmos Policy:

```sh
cd ../cosmos-spike
uv run cosmos-convert-native ../viam-isaac-sim/runs/picks-001 datasets/picks-001 \
  --action-space joint_absolute
uv run cosmos-convert-native ../viam-isaac-sim/runs/picks-001 \
  datasets/picks-001-with-cosmoswriter --action-space joint_absolute \
  --include-cosmoswriter
```

Successful episodes are selected by default. `--outcomes success,failure,incomplete`
packages all finalized outcomes for world/value experiments. Each conversion
contains both 7-D absolute joint actions and 10-D relative end-effector actions;
`--action-space` chooses which is exposed through the canonical `action` feature.
Both variants retain the canonical fixed/wrist RGB videos, states, tasks, and
native action representations. The opt-in variant adds only auxiliary
CosmosWriter controls. `cosmos-prepare-droid` creates a separate job-local
DROID derivative that strips those auxiliary controls; it does not alter the
primary direct dataset.

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

Isaac Sim is isolated in the `full` project extra. Module setup and launch use
`uv sync --extra full` / `uv run --extra full`; CPU-only development and CI can
use the default `uv sync` without downloading Isaac Sim.

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

Before a GPU data-generation run, exercise the complete headless UR/Robotiq
cell with the Isaac interpreter:

```sh
PYTHONPATH=src "$ISAAC_PYTHON" examples/l4_smoke.py
```

Success prints one JSON line with `"status":"PASS"`, six arm joints, a
normalized gripper position, and `640x360` frames from both cameras.

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

## Running policies

- in `../cosmos-spike`, start `scripts/launch_ur20_policy_server.py`
- in `../arm-policy`, run `examples/policy_workflow.py` with the desired prompt

## Status / roadmap

- [x] world boot, stage loading, livestream, reset, seeded prop randomization/query
- [x] arm joint control (UR family, Franka, arbitrary USD articulations)
- [x] RGB cameras
- [x] differential-drive bases
- [x] cloud builds / registry publishing (tag a release)
- [x] serve kinematics files (`GetKinematics`) so Viam's motion service can do
      IK and planning for simulated arms (all motion stays in Viam, not Isaac)
- [ ] depth / point clouds from cameras
- [x] UR-mounted Robotiq 2F-85 gripper support
