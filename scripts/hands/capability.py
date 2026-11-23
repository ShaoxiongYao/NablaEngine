"""The forest engine's capability report of MuJoCo Menagerie's hands, as they are and after envs.convert.to_engine,
each alone and with a free cube. CPU only (no GPU, no nvcc).

  PYTHONPATH=src python scripts/hands/capability.py --menagerie <mujoco_menagerie checkout> [--out runs/hands/capability.json]

Exit status 0 when every converted model has no capability error, 1 otherwise.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import common
import nablaengine
from nablaengine.envs import convert
from nablaengine.model import capability


def sizes(m):
    return dict(nbody=int(m.nbody), nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), ngeom=int(m.ngeom), npair=int(m.npair),
                ntendon=int(m.ntendon), neq=int(m.neq), nmesh=int(m.nmesh))


def pair_types(m):
    if not m.npair:
        return {}
    t, n = np.unique(np.concatenate([m.geom_type[m.pair_geom1], m.geom_type[m.pair_geom2]]), return_counts=True)
    return {mujoco.mjtGeom(int(a)).name[len("mjGEOM_"):].lower(): int(b) for a, b in zip(t, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", required=True)
    ap.add_argument("--hands", default=",".join(common.HANDS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"nablaengine {nablaengine.__version__}, mujoco {mujoco.__version__}")
    res, ok = {}, True
    for hand in args.hands.split(","):
        for cube in (False, True):
            key = hand + ("+cube" if cube else "")
            xml = common.mjcf(args.menagerie, hand, cube)
            raw = mujoco.MjModel.from_xml_string(xml)
            r0 = capability.check_forest(raw)
            print(f"\n=== {key}\nsource    {sizes(raw)}\n          {r0.summary()}")
            for f in r0.errors():
                print("          " + f.line()[:240])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                c = convert.to_engine(xml)
            r1 = capability.check_forest(c.model)
            print(f"converted {sizes(c.model)}\n          pair geom types {pair_types(c.model)}\n          {r1.summary()}")
            for f in r1.errors() + r1.warnings():
                print("          " + f.line()[:240])
            for n in c.notes:
                print("          note: " + str(n)[:240])
            ok &= r1.ok()
            res[key] = dict(source=dict(sizes(raw), **r0.to_dict()), converted=dict(sizes(c.model), **r1.to_dict()),
                            pair_types=pair_types(c.model), notes=[str(n) for n in c.notes],
                            warnings=[str(x.message) for x in w])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(nablaengine=nablaengine.__version__, mujoco=mujoco.__version__,
                                                  models=res), indent=1))
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
