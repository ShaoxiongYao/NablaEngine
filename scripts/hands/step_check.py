"""One hand with a free cube on the forest engine (JAX): build it, then one control step of states from MuJoCo C
rollouts against MuJoCo C in float64, and the gradient of an output with respect to ctrl (finite or not).

  source .dk_env/activate.sh
  python scripts/hands/step_check.py --menagerie <mujoco_menagerie checkout> --hand allegro [--out runs/hands/allegro.json]

One model per process (jax_ffi.init holds one library). Needs a GPU and nvcc; the build takes host memory (a 14 GB
laptop ran out of it on the Allegro build), so watch `free -g` on the first run. Not a gate yet: it prints and
stores the numbers, exit status 0 unless an output is not finite.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import common


def corpus(m, nworld, substeps, seed):
    """States of MuJoCo C rollouts under random position targets (the cube falling onto or past the hand), and
    their successors one control step (substeps physics steps) later under a perturbed target."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = {k: [] for k in ("qpos", "qvel", "warm", "ctrl", "qpos1", "qvel1", "ncon")}
    for _ in range(nworld):
        mujoco.mj_resetData(m, d)
        u = lo + (hi - lo) * rng.uniform(0.2, 0.8, m.nu)
        d.ctrl[:] = u
        for _ in range(int(rng.integers(20, 200))):
            mujoco.mj_step(m, d)
        u = np.clip(u + 0.1 * (hi - lo) * rng.standard_normal(m.nu), lo, hi)
        d.ctrl[:] = u
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("warm", d.qacc_warmstart), ("ctrl", u)):
            out[k].append(np.array(v))
        out["ncon"].append(int(d.ncon))
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        out["qpos1"].append(d.qpos.copy())
        out["qvel1"].append(d.qvel.copy())
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", required=True)
    ap.add_argument("--hand", required=True, choices=list(common.HANDS))
    ap.add_argument("--cube-pos", default=None)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-contacts", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from nablaengine import engines, jax_ffi
    from nablaengine.envs import convert

    m = convert.to_engine(common.mjcf(args.menagerie, args.hand, cube=True, cube_pos=args.cube_pos)).model
    res = dict(hand=args.hand, nv=int(m.nv), nu=int(m.nu), npair=int(m.npair), timestep=float(m.opt.timestep),
               engine=engines.name(m))
    print(res, flush=True)
    t = time.time()
    info = jax_ffi.init(model=m, foot_pairs=[], max_contacts=args.max_contacts, max_rows=args.max_rows)
    res.update(init_seconds=round(time.time() - t, 1), built=bool(info.get("built")), maxcon=info.get("maxcon"),
               nefc=info.get("nefc"), arch=info.get("arch"), substeps=jax_ffi.SUBSTEPS, iterations=jax_ffi.ITERATIONS)
    print(f"init: {res['init_seconds']} s, built {res['built']}, maxcon {res['maxcon']}, nefc {res['nefc']}", flush=True)

    c = corpus(m, args.nworld, jax_ffi.SUBSTEPS, args.seed)
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    q, v, w, u = (f32(c[k]) for k in ("qpos", "qvel", "warm", "ctrl"))
    friction = jnp.ones((args.nworld,), jnp.float32)  # no foot pairs: unused
    com = jnp.zeros((args.nworld, int(info["com_words"])), jnp.float32)
    q1, v1, _ = (np.asarray(x) for x in jax_ffi.step(q, v, w, u, friction, com))
    rel = lambda a, b: np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    ev, eq = rel(v1, c["qvel1"]), rel(q1, c["qpos1"])
    g = np.asarray(jax.grad(lambda x: jnp.sum(jax_ffi.step(q, v, w, x, friction, com)[1] ** 2))(u))
    finite = bool(np.isfinite(q1).all() and np.isfinite(v1).all())
    res.update(nworld=args.nworld, finite=finite,
               ncon=dict(min=int(c["ncon"].min()), median=float(np.median(c["ncon"])), max=int(c["ncon"].max())),
               qvel_rel=dict(median=float(np.median(ev)), p90=float(np.quantile(ev, 0.9)), max=float(ev.max())),
               qpos_rel=dict(median=float(np.median(eq)), p90=float(np.quantile(eq, 0.9)), max=float(eq.max())),
               grad_ctrl=dict(finite=bool(np.isfinite(g).all()), norm=float(np.linalg.norm(g))))
    try:
        res["diagnostics"] = {k: np.asarray(x).tolist() for k, x in jax_ffi.read_diagnostics().items() if np.size(x) <= 8}
    except Exception as e:  # the diagnostics' form was not checked when this script was written
        res["diagnostics"] = f"{type(e).__name__}: {e}"[:300]
    for k in ("ncon", "qvel_rel", "qpos_rel", "grad_ctrl", "diagnostics"):
        print(f"{k}: {res[k]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
