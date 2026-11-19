# uf850_g2.usd

A UFACTORY 850 arm with a UFACTORY Gripper G2, as one articulation.

## Source

Built from UFACTORY's official robot description, `xarm_description` in
[xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2)
(branch `humble`, commit `ac736c9`), expanded with
`robot_type:=uf850 dof:=6 add_gripper:=true gripper_version:=G2`, then converted
with NVIDIA's `urdf-usd-converter` 0.4.0.

## Changes from the source

- Links placed directly under the robot prim, with the articulation root on it.
- G2 linkage: the five follower joints are PhysX mimic joints of `drive_joint`
  (the description's mimic tags), so only `drive_joint` is driven. 255 solver
  position iterations keep the pads parallel under grip load.
- G2 range set to a measured real G2: 95 mm between the pads when open, 0 mm
  closed (UFACTORY's limits give about 89 mm to 1.7 mm).
- `drive_joint` force and speed caps from the Viam driver's G2 defaults (force 50,
  speed 2000; about 40 N and 109 mm/s by UFACTORY's force chart and SDK).
- Box colliders on the finger pads; a stray fragment about 0.5 m from the
  gripper removed from the G2 base mesh.
- Gravity off on the robot links (the real arm holds its commanded joints),
  position drives on joint1 to joint6.

## License

The meshes and kinematic description are UFACTORY's, under this license:

```
Copyright (c) 2018, UFACTORY Inc.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
