[Watch the project demonstration on Bilibili](https://www.bilibili.com/video/BV1MFbK6zEG5/?)

> **Source-code release:** This repository includes source code, configuration files, tests, and documentation. Pretrained checkpoints, experiment logs/results, plots, videos, and binary mesh assets are not included. Train a policy before running checkpoint-based evaluation or deployment. The Webots visual model references `webots_alvik/meshes/alvik_reference_design.stl`, which must be supplied separately.

# SAC / CBF-QP for the real Arduino Alvik

Implementation of the experiment defined in [PLAN.MD](PLAN.MD). Three methods are
compared under one identical setup; the only variable is how the CBF-QP
correction term `delta_u = u_safe - u_nom` is coupled to SAC.

| mode | name in PLAN.MD | executed action | critic current action | Bellman next / actor | gradient through QP |
|---|---|---|---|---|---|
| `vanilla` | M1 | `u_nom` | `u_nom` | `pi(s')`, `pi(s)` | no QP at all |
| `external_qp` | M2 | `u_safe` | `u_safe` | `pi(s')`, `pi(s)` | cut to zero |
| `diff_qp` | M3 | `u_safe` | `u_safe` | `QP(s', pi(s'))`, `QP(s, pi(s))` | flows through |

`external_qp` and `diff_qp` call the *same* QP function; the only difference is
whether the result is detached. `tests/test_sac.py` asserts that with no active
constraints the two produce bit-comparable updates, so any future divergence
between them can only come from the safety-layer coupling.

## Install and check

```bash
D:\Anaconda\envs\pytorch\python.exe -m pip install -r requirements.txt
```

```bash
D:\Anaconda\envs\pytorch\python.exe -m unittest discover -s tests -t . -v
```

Environment used for development: Python 3.9.21, numpy 1.26.4, torch 1.12.0+cu116
(CUDA available), scipy 1.13.1.

## Gates

The gates from PLAN.MD 11.2, with the command for each.

**G0 - unit tests.** The command above. 138 tests covering the ToF sign
convention, the endpoint memory, QP optimality against a grid search, QP
gradients against finite differences, the three-method semantics, the invisible
map border, and the scenario generator.

**G1 - smoke, all three methods.**

```bash
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --config configs/smoke.json --modes vanilla external_qp diff_qp --seeds 42 --device cpu --run-root runs_smoke --name smoke --evaluate
```

**G2 - obstacle-free navigation acceptance. Blocking gate: do not start the
main training until this passes.**

```bash
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --config configs/goal_only_debug.json --modes vanilla diff_qp --seeds 42 --steps 50000 --device cuda --evaluate
```

Pass condition: held-out success >= 0.80 and mean final true distance <= 0.07 m.
If it fails, the fix is in the reward scale, the action mapping, the observation
normalisation, the discount or the entropy floor - not in more steps. The two
earlier attempts in `my_project_with_real_Alvik*` failed exactly here and were
answered with more steps, which did not help.

**G3 - timing benchmark.**

```bash
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --config configs/main_80cm.json --modes vanilla external_qp diff_qp --seeds 42 --steps 3000 --device cuda --run-root runs_bench --name timing
```

Measured on this machine (RTX-class GPU, 8 CPU threads, torch 1.12+cu116), with
the per-training-step cost isolated by a two-point run so that setup and
validation cancel:

| mode | train step | 200k training | 8 validations | total per run |
|---|---:|---:|---:|---:|
| `vanilla` | 10.93 ms | 36.5 min | 2.3 min | **38.7 min** |
| `external_qp` | 13.55 ms | 45.2 min | 6.1 min | **51.2 min** |
| `diff_qp` | 18.86 ms | 62.9 min | 6.1 min | **69.0 min** |

So G4 (3 methods x 5 seeds) is about **13 hours sequential**, and that is an
upper bound: validation episodes shorten from 300 steps towards ~150 as the
policy starts reaching the goal. The safety layer costs ~2.6 ms per step in
rollout, and differentiating through it costs a further ~5.3 ms per step for the
two batched QP calls in every update.

Training is faster on GPU than CPU (`vanilla` 10.9 ms vs 16.3 ms per step), so
the device split stands: batched updates on GPU, single-sample rollouts on CPU.
Per-step cost is launch-bound rather than FLOP-bound, so running the three modes
as three concurrent processes on one GPU is close to free and cuts G4 to roughly
5 hours.

**G4 - main experiment, 3 methods x 5 seeds x 200k steps.**

```bash
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --config configs/main_80cm.json --modes vanilla external_qp diff_qp --seeds 42 43 44 45 46 --steps 200000 --device cuda --evaluate
```

**G5 - aggregation and statistics.**

```bash
D:\Anaconda\envs\pytorch\python.exe summarize_experiments.py --root runs --output results/main_80cm
```

**Deployment export** (only `diff_qp`; the exporter refuses other modes):

```bash
D:\Anaconda\envs\pytorch\python.exe export_policy.py --checkpoint runs/diff_qp/RUN/checkpoints/best.pt
```

`--steps` rescales every step-based schedule (curriculum boundaries, validation
interval, checkpoint interval, random pre-fill) so a short run exercises the same
code path. The main experiment must be run at the config's own budget.

## Layout

```
safe_alvik/config.py           defaults + consistency checks (rejects a look-ahead
                               double-count, map-boundary handling, reverse motion)
safe_alvik/geometry.py         body frame, unicycle integration, action mapping,
                               inverse-gain command compensation
safe_alvik/tof_model.py        four-zone ray casting, calibrated bias/noise,
                               three-frame median, endpoint extraction
safe_alvik/obstacle_memory.py  8 cm clustering, 4 slots, 1.4 s TTL, quadrant features
safe_alvik/cbf_qp.py           batched differentiable QP by active-set enumeration
safe_alvik/environment.py      2-D environment, scenario generator, disturbances
safe_alvik/networks.py         actor + twin critic (identical for all methods)
safe_alvik/replay.py           stores nominal and safe actions plus constraint rows
safe_alvik/sac.py              the three couplings - the only place they differ
safe_alvik/trainer.py          curriculum with promotion gates, validation, checkpoints
safe_alvik/evaluation.py       held-out evaluation, metrics, model selection
```

## Design notes worth knowing before reading the code

**The QP is solved by enumeration, not iteration.** With two decision variables
the optimal active set has size 0, 1 or 2, so all 37 candidate optima can be
written in closed form and evaluated at once. The feasible candidate with the
lowest objective is the global optimum, and autograd through the selected
candidate's closed form is exactly the implicit-function gradient. This is what
makes the layer batched, fast and differentiable without an inner solver.

**Rollouts run on CPU, batched updates on GPU.** Single-sample inference is
kernel-launch bound: measured here, one actor forward costs 0.157 ms on CPU
against 0.554 ms on CUDA, and one QP solve 0.64 ms against 2.30 ms, while a
batch-256 QP is the other way round (2.42 ms CPU, 1.74 ms CUDA). The QP layer is
stateless, so `SACAgent` simply keeps two copies. Validation rollouts use
`agent.eval_runner()`, a frozen CPU copy of the actor built once per validation;
`tests/test_sac.py` asserts it reproduces the agent's action exactly.

**Infeasibility is rare by construction.** Stopping satisfies every row whenever
`alpha * h_i >= delta_v`, i.e. whenever the robot is more than 6 mm outside the
margin. When the hard problem really is infeasible the fallback forces `v = 0`
but keeps `omega`: braking without being allowed to turn away was the lock-up
observed on the real robot.

**The endpoint bearing is only known to +/- 7.5 degrees.** A zone reports a range,
not a direction, so an endpoint is placed on the zone centre line and can be up
to `0.131 d` from the true surface point - about 3.3 cm at the distances where
the QP intervenes. The 30 mm obstacle margin is sized to absorb this. It is one
reason this is an engineering CBF, not a formal guarantee.

**The endpoint memory TTL is 5.0 s, not the 1.4 s in `parameter_opus.txt`.**
That value came from a project running at 0.25 m/s, where it covered 35 cm of
travel; at 3 cm/s it covers 4.2 cm, less than one body radius, so an obstacle
leaving the 60 degree FoV is forgotten before the robot has driven past it.
Measured with a random policy on 200 held-out episodes
(`diagnostics/qp_memory_sweep.py`):

| setting | collisions | episodes with clearance < 0 | QP infeasible steps |
|---|---:|---:|---:|
| no safety layer | 23.0% | 38.0% | - |
| QP, ttl 1.4 s | 10.0% | 30.5% | 3.66% |
| QP, ttl 3.0 s | 6.0% | 26.0% | 4.82% |
| **QP, ttl 5.0 s** | **1.5%** | **16.5%** | 7.80% |
| QP, ttl 8.0 s | 1.5% | 13.0% | 10.19% |

The safety layer clearly works, and just as clearly does not eliminate
collisions: the forward-only FoV means a turned-away obstacle exists only in
memory. `max_tracks` was checked the same way; 4 is enough (4/6/8 all give 1.5%).

**A longer TTL also keeps phantom obstacles alive longer, so tracks need
confirming.** On a map with *no obstacles at all*, the QP still had an active
constraint on 14.7% of steps. A single false low is filtered by the 3-frame
median, but two inside one window are not (p ~ 0.0012 per frame per zone, about
one per episode), and the resulting phantom then survives the whole 5 s TTL.
This is why `vanilla` and `diff_qp` are not equivalent on an obstacle-free map
even though the QP "should" be a pass-through there. A track now has to be seen
in `memory.confirm_frames` separate frames before it constrains anything - the
same idea as the board's 3-consecutive-sample rule - and unconfirmed tracks
cannot evict confirmed ones:

| confirm_frames | phantom steps (empty map) | collisions | episodes with clearance < 0 | QP infeasible |
|---:|---:|---:|---:|---:|
| 1 | 14.68% | 3.0% | 22.0% | 7.56% |
| **2 (default)** | **9.95%** | 2.0% | 15.5% | 7.32% |
| 3 | 3.34% | 3.0% | 17.5% | 3.17% |

Phantom rate and infeasibility fall monotonically; the collision differences are
inside the noise at n=200 (standard error ~1.2%) and should not be read as a
ranking.

**The odometry's +2.7% distance overestimate is corrected on the PC, not left
in the noise.** Calibration runs R01-R04 report 99.65 mm of odometry for a
100 mm command while the map-measured truth was ~97 mm. Over the 0.735 m task
that bias is 19.8 mm in one direction - exactly the whole budget between the
50 mm stop radius and the 70 mm success radius. So `robot.odom_distance_gain`
= 1.027 is divided out on the PC side, the same pattern already used for the
drive-gain compensation, and **the real-robot runtime must do the same**
(it is carried in the `export_policy.py` metadata). The heading random walk was
also 7x too pessimistic against the same measurements (0.3 deg/step, ~3.9 deg
over the task, versus a measured 0.076 deg/100 mm, ~0.56 deg) and is now
0.05 deg/step. End-of-run odometry error went from 31.6 mm mean / 59.8 mm p95
to 12.4 mm / 26.3 mm, with the signed along-track bias falling from +19.8 mm to
+0.6 mm. Three tests in `tests/test_environment.py` pin this.

**The board safety layer is deliberately NOT simulated.** The real board blocks
forward motion below 100 mm and brakes below 60 mm. Modelling it during training
would mask `vanilla` collisions and destroy the safety comparison. It belongs in
the real-robot runtime, where its trigger count is recorded as a proxy for the
CBF layer having failed to act in time.

**Time limits are bootstrapped, collisions and goal stops are not.** A 300-step
cut-off is not a terminal state of the task; treating it as one teaches the
value function that the world ends after 60 s.

**Model selection is lexicographic**: success up, collision down, final true
distance down, return up. Ranking on collisions or return first lets a policy
that never moves win, which is what happened in the previous project.
