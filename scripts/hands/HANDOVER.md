# Handover: dexterous hands on the forest engine

Written 2026-11-23 on a laptop (RTX 4060 Laptop, 14 GB RAM) for whoever continues on the RTX 4090 desktop. This file
is a working note for the branch `hand-checks`; remove it before any pull request to upstream.

## Goal

Use NablaEngine for dexterous in-hand manipulation: a fixed-base hand moving a free object. The owner of this fork
(@ShaoxiongYao) proposed it upstream in issue #2; the maintainers closed it saying 1.0.0's forest engine covers both
blockers (fixed-base trees, free objects) and asked for an issue if a hand model is still refused. The task now is to
find out whether the hands really build, run and match MuJoCo, and to turn that into either a contribution (a hand
model with its checks) or a precise issue.

## State of the repository

- `main` = `origin/main` = `upstream/main` = `63618b7` (NablaEngine 1.0.2). Keep `main` a mirror of upstream.
- `origin`: the fork, `https://github.com/ShaoxiongYao/NablaEngine.git`. `upstream`:
  `https://github.com/sailab-ai/NablaEngine.git` (it moved from `OamIcs/NablaEngine`; a fresh clone of the fork has
  no `upstream` remote, add it).
- This branch, `hand-checks`, is `upstream/main` plus `scripts/hands/`.

## What is established

Measured with nablaengine 1.0.2, MuJoCo 3.17.0, MuJoCo Menagerie at `98a70c569c4c1b6da467a20012f766ee1f8dc521`
(`scripts/hands/capability.py`, CPU only, reruns in a few seconds):

- Menagerie's `right_hand.xml` of LEAP, Allegro and Shadow, as they are: each refused by `capability.check_forest`
  with exactly two codes, T3 (jointless bodies: the palm, Allegro's four tips, Shadow's forearm) and H15 (contype /
  conaffinity contacts).
- After `nablaengine.envs.convert.to_engine`: 0 errors and 0 warnings for all three, alone and with a free cube.

| model | nv | nu | pairs | pair geom types |
|---|---|---|---|---|
| leap + cube | 22 | 16 | 1692 | box, mesh |
| allegro + cube | 22 | 16 | 206 | capsule, box |
| shadow + cube | 30 | 20 | 590 | sphere, capsule, cylinder, box, mesh; 4 fixed tendons |

## What is not established

- **No hand has been built or stepped.** `scripts/hands/step_check.py` has never run to completion: the first
  attempt stopped at a missing `foot_pairs` (fixed: `foot_pairs=[]`, as `nablaengine/envs/env.py` does), the next two
  were cut off when the laptop ran out of host memory during the Allegro build (kernel OOM killer, 14 GB RAM). So
  everything after `jax_ffi.init` in that script is untested code: expect to fix small things (the diagnostics
  block is a guess at `read_diagnostics()`'s form).
- Whether the build's memory use was the model's size or the laptop's other load is unknown. Check `free -g` before
  and during the first build.
- The cube positions in `common.py` are guesses; the hands' palm orientations were not looked at, so the cube may
  miss the hand. `step_check.py` prints the contact counts of the corpus: if they are near 0, place the cube
  (`--cube-pos`) so that it rests on the hand.
- LEAP has about 1700 pairs against a default capacity of 64 contacts / 192 rows per world (forest maximum 255 /
  1023, `init(max_contacts=, max_rows=)`): build time and capacity overflow are the likely trouble there.

## Setting up on the desktop

```bash
git clone https://github.com/ShaoxiongYao/NablaEngine.git && cd NablaEngine
git remote add upstream https://github.com/sailab-ai/NablaEngine.git
git fetch upstream --tags && git switch hand-checks
scripts/setup_env.sh --dry-run && scripts/setup_env.sh    # .dk_env/, a few GB; picks nvcc from the driver version
source .dk_env/activate.sh
scripts/verify.sh --quick                                 # the installation, a few minutes
git clone https://github.com/google-deepmind/mujoco_menagerie ../mujoco_menagerie
git -C ../mujoco_menagerie checkout 98a70c569c4c1b6da467a20012f766ee1f8dc521
```

## Next steps, in order

1. `python scripts/hands/capability.py --menagerie ../mujoco_menagerie`: must print PASS, as on the laptop.
2. `python scripts/hands/step_check.py --menagerie ../mujoco_menagerie --hand allegro --out runs/hands/allegro.json`
   (the smallest model). Look at: the build time, contacts in the corpus, qvel / qpos relative error against MuJoCo
   C, a finite gradient. For scale, the repository's gates for the G1 are qvel median 5e-6, p99 5e-5
   (`docs/verification.md`); MuJoCo C here runs its default solver settings, not the tight float64 truth.
3. The same for `--hand shadow`, then `--hand leap` (one model per process).
4. Decide with the owner what to send upstream:
   - a model is refused or wrong: an issue with the code or the numbers and the exact commands;
   - they work: a hand-with-object model and its check in the form of the accepted PR #3 (a fixture under
     `fixtures/robots/<name>/`, export and check scripts, docs, `CHANGELOG.md`, licences in
     `THIRD_PARTY_NOTICES.md`), validated with `scripts/robots/robot_check.py` against MuJoCo Warp where it applies.

## How upstream takes contributions

There is no CONTRIBUTING file; this is from the history. Upstream develops privately and publishes each release as
one squashed commit, so engine changes easily collide with unpublished work: PR #1 (H1 support) was superseded by
0.2.0, and the plan of issue #2 by 1.0.0. Open an issue before touching `src/nablaengine/`. Self-contained additions
(models, checks, docs) are what was taken as is (PR #3: the commit went to `main` with its author, the PR itself was
closed). There is no test CI: say in the PR what was run and what was not (`scripts/verify.sh`), and rebase on
`upstream/main` often, since each release rewrites a lot.

## Working rules of the owner

- Ask before pushing or opening issues / PRs; nothing has been posted upstream from this work.
- Answers in Chinese.
