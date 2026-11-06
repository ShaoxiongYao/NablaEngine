# The H1 model (`fixtures/h1`)

`h1_train.mjb` is the Unitree H1 humanoid for the engine: a floating base and 19 hinge joints, each driven by a
position (PD) actuator, on a flat floor. It is built from the Unitree H1 description in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) by `scripts/make_h1_model.py`, which applies
the recipe that turned Menagerie's G1 into the mjlab G1 (`fixtures/mjlab/README.md`).

`nablaengine.fixtures.load_model(fixtures.H1)` loads the model and checks that the engine supports it; the JAX and
PyTorch interfaces take it as `init(model=m, foot_pairs=meta["foot_pairs"])`. `scripts/h1_check.py` (in
`scripts/verify.sh`) checks the engine on it against MuJoCo C in float64.

| File | Content |
|---|---|
| `h1_train.mjb` | The compiled model in MuJoCo's binary format, 142655 bytes, sha256 `397dabd9e31f284133533d548b45eee96b4f4697b27febc1a80eee7ccd553ae9`. |
| `h1_meta.json` | Metadata of the model, with the same keys as `fixtures/mjlab/g1_meta.json` (`robot`: `"h1"`). |
| `LICENSE` | The licence of the Unitree H1 description (BSD 3-Clause), copied verbatim from MuJoCo Menagerie. |

## The model

| | |
|---|---|
| Degrees of freedom | `nq` 26, `nv` 25: a free root joint and 19 hinge joints. 21 bodies (the world included), 19 actuators. |
| Collision geometry | 33 geoms: the floor plane, 6 spheres and 26 capsules, 3 of them on each sole. No meshes, boxes or other geom types. |
| Contacts | 457 explicit contact pairs and nothing else; the 6 sole-floor pairs are `foot_pairs` (per-world friction). |
| Options | Timestep 5 ms, Newton solver, pyramidal friction cones, `implicitfast` integrator, Euler damping disabled. |
| Actuators | kp 394.8, kd 25.1 on every joint (mjlab's 10 Hz, damping ratio 2, from Menagerie's armature 0.1); Menagerie's torque limits as the joints' actuator force ranges. |
| Control step | 4 substeps (20 ms). |
| Other content | The keyframe `home`, a light, the IMU site; `torso_link` takes the per-world COM offset. |

With `nv` 25 the H1 runs the engine's solves with the lanes past `NV` masked (capability warning H3) and the register
Cholesky instead of the G1's tuned broadcast variant (H5).

Joint PD alone does not keep the H1 standing: its centre of mass is about 2 cm in front of the heels at the home
keyframe, and it tips backwards within about two seconds, as Menagerie's model does under the same gains. A demo
that keeps it up optimizes the setpoints (`examples/squat_trajopt.py --robot h1`).

## How it was derived

The source is MuJoCo Menagerie at commit
[`98a70c569c4c1b6da467a20012f766ee1f8dc521`](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/unitree_h1),
file `unitree_h1/scene.xml` (which includes `unitree_h1/h1.xml`). The steps are those of `scripts/make_h1_model.py`
(its docstring lists them): cylinders and the torso box replaced by capsules, explicit contact pairs as in mjlab's G1
contact set (minus three pairs whose capsules touch at the home keyframe), meshes, materials and textures removed,
5 ms and `implicitfast`, position servos with mjlab's gains and setpoint range, no joint damping, MuJoCo's default
joint-limit parameters. With `mujoco==3.17.0` the script reproduces `h1_train.mjb` byte for byte:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie && git -C mujoco_menagerie checkout 98a70c5
python scripts/make_h1_model.py --menagerie mujoco_menagerie --out fixtures/h1
```

## Licences

- **Unitree H1 description** (MuJoCo Menagerie `unitree_h1`): Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY
  CO.,LTD. ("Unitree Robotics"), BSD 3-Clause License. Its full text is `LICENSE` in this directory, and it applies to
  `h1_train.mjb`, a modified binary form of that description.
- **mjlab**: Apache License 2.0, Copyright 2025, The mjlab Developers. The conversion follows mjlab's G1
  configuration (the contact set, the contact and joint parameters, the PD gain design and the setpoint range); it
  contains no mjlab code.
- The other files of this directory are covered by the repository's licence (Apache-2.0).

Unitree Robotics, Google DeepMind and the mjlab developers do not endorse this project. Their names are used only to
identify the sources.
