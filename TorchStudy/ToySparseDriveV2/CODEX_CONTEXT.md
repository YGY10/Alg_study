# ToySparseDriveV2 Codex Context

For any task under `TorchStudy/ToySparseDriveV2`, read `TorchStudy/ToySparseDriveV2/CODEX_CONTEXT.md` first.

This file is the handoff context for Codex. Read this file first before editing
or analyzing `TorchStudy/ToySparseDriveV2`.

## Project Goal

`ToySparseDriveV2` is a study project that moves `ToyTrajectorySelector` closer
to real SparseDriveV2 behavior.

The project uses real SparseDriveV2 vocab files:

- path vocab: `SparseDriveV2/ckpt/kmeans/path_1024.npy`, shape `[1024, 50, 3]`
- velocity vocab: `SparseDriveV2/ckpt/kmeans/velocity_256.npy`, shape `[256, 8]`
- trajectory vocab: `SparseDriveV2/ckpt/kmeans/trajectory_1024_256.npz`, shape `[1024, 256, 8, 3]`

The toy task is not to regress arbitrary trajectories. The model selects:

```text
path index + velocity index -> trajectory from real SparseDriveV2 vocab
```

## Coordinate Convention

Use the vehicle/world convention:

```text
x: forward
y: left
```

Plotting uses:

```text
plot x-axis = y left
plot y-axis = x forward
```

Many plots invert the x axis with:

```python
ax.set_xlim(grid_config.y_max, grid_config.y_min)
```

Do not casually change coordinate signs. This was previously a source of confusion.

## Important Files

- `vocab/vocab.py`
  Loads real SparseDriveV2 vocab and provides summary/coverage helpers.

- `dataset/grid.py`
  Grid/world conversion and raster drawing helpers.

- `dataset/dataset.py`
  Synthetic dataset. It samples a route path from the path vocab, uses the route
  endpoint as `goal_xy`, samples dynamic obstacles, calls the teacher, and returns
  model inputs plus teacher supervision.

- `dataset/teacher.py`
  Multi-objective teacher candidate scorer. It scores paths and trajectories
  using goal, route, progress, clearance, collision, comfort, speed, initial
  acceleration, acceleration, and jerk.

- `models/model.py`
  CNN scene encoder + ego encoder + path scorer + velocity scorer + trajectory
  scorer. The model can score mixed trajectory candidates from model top-k paths
  plus teacher top-k paths.

- `losses/losses.py`
  Training losses. Current main loss is teacher-distribution imitation plus
  explicit collision margin loss.

- `train.py`
  Main training loop. It performs online teacher scoring for the model's current
  candidate set.

- `test_random.py`
  Manual/random evaluation. Supports custom obstacles, custom ego state, and
  custom goal point.

## Current Data Generation

Current training constants in `train.py`:

```python
NUM_SAMPLES = 1024
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
```

Dataset logic:

```python
route_path_index = random path vocab index
route_path = vocab.path[route_path_index]
goal_xy = route_path[-1, :2]
```

So `goal_xy` is random through the randomly selected path endpoint. It is not
sampled independently in continuous space.

Current obstacle sampling is mixed:

```text
num obstacles: 4 to 12
70% route-aware obstacles
30% global random obstacles
```

Global random obstacles use:

```text
x: grid_config.x_min + 8 to grid_config.x_max - 8
y: grid_config.y_min + 8 to grid_config.y_max - 8
length: 3 to 8
width: 2 to 5
vx: -4 to 4
vy: -3 to 3
```

Route-aware obstacles are sampled near the selected route:

```text
route anchor x: 10m to 45m forward
center_x = route_x + uniform(-3m, 3m)
center_y = route_y + uniform(-10m, 10m)
```

This was added to increase route-adjacent hard collision-boundary cases.

## Teacher Strategy

`dataset/teacher.py` has a two-stage scorer:

1. Score all 1024 path vocab entries.
2. Keep `TeacherConfig.num_path_candidates` paths, then score all 256 velocity
   choices for each kept path.

Default dataset teacher config:

```python
TeacherConfig(
    num_path_candidates=8,
    num_top_trajectories=8,
    temperature=2.0,
)
```

Important teacher cost terms:

```text
collision_weight = 1e6
path_collision_weight = 1000
goal_weight
route_weight
clearance_weight
progress_weight
comfort_weight
speed_weight
initial_accel_weight
accel_weight
jerk_weight
```

Teacher can miss good paths because path-stage pruning is only top-8. We have
seen cases where model-selected paths outside teacher top-8 have lower full
trajectory cost than the teacher reference. Therefore, do not judge correctness
only by whether `pred_in_teacher_top_candidate_set` is true. Judge by collision
and teacher cost.

## Model Inputs

Model currently receives:

```text
input_grid: 4-channel raster
ego_state: [speed, ego_length, ego_width]
path_vocab
velocity_vocab
trajectory_vocab
```

`input_grid` channels:

```text
0: current obstacles
1: future obstacle swept area
2: route path
3: goal point
```

The teacher also uses goal, route, obstacles, and ego state. The model sees those
same signals through raster channels and ego vector.

## Candidate Set During Training

During training, model trajectory candidates are:

```text
model top-k path + teacher top-k path
then expanded over all 256 velocity vocab entries
```

The relevant model forward argument is:

```python
extra_path_indices=batch["teacher_path_indices"]
```

This means the candidate set includes both model-generated candidates and teacher
candidates. This is important because training must punish bad model-generated
candidates.

## Current Loss

The current total loss is:

```text
loss = path_loss
     + trajectory_loss
     + 0.5 * collision_margin_loss
```

Velocity loss is computed for logging but its weight is currently `0.0`, so
`vel_acc` can be low and should not be treated as the main signal.

### path_loss

Soft cross entropy:

```text
path_scores [B, 1024] vs teacher_path_probs [B, 1024]
```

Teacher path probs are derived from teacher path costs.

### trajectory_loss

The training loop calls teacher online on the actual model candidate set:

```python
build_teacher_candidate_targets_for_batch(...)
```

It returns:

```python
{
    "probs": candidate teacher probability,
    "costs": candidate teacher cost,
    "collision": candidate collision mask,
    "clearance": candidate clearance,
}
```

Then:

```text
trajectory_scores [B, C] vs teacher_candidate_probs [B, C]
```

This is better than old L2-to-target supervision because teacher cost includes
collision, clearance, goal, route, and comfort terms.

### collision_margin_loss

Explicit safety ranking loss. It only applies to samples that contain both safe
and collision candidates.

Goal:

```text
collision candidate score <= best safe candidate score - margin
```

Implementation:

```text
relu(collision_score - best_safe_score + margin)
```

Defaults:

```python
collision_margin = 2.0
collision_margin_weight = 0.5
```

This was added because soft teacher-distribution loss was not enough to prevent
some collision candidates from receiving high trajectory scores.

Important bug already fixed:

Do not use `-inf` and then multiply invalid rows by zero. That caused `nan` in
`collision_margin_loss`. Current implementation filters valid rows first and
uses `torch.finfo(dtype).min`.

## Training Metrics

Current log fields:

```text
loss
path
vel
traj
margin
path_acc
vel_acc
topk_path
model_topk
traj_vel
traj_pair
traj_l2
end_l2
collision
safe
pred_cost
```

Interpretation:

- `topk_path`: whether final mixed candidate set contains teacher path. This is
  almost always 1.0 because teacher paths are injected.
- `model_topk`: whether the model's own top-k path set contains the teacher best
  path. This is the real path branch recall metric.
- `traj_vel`: selected trajectory velocity matches teacher velocity.
- `traj_pair`: selected path+velocity exactly matches teacher reference pair.
- `traj_l2`, `end_l2`: geometric distance to teacher reference. Useful but not a
  safety metric.
- `collision`: collision rate of the model-selected final trajectory.
- `safe`: safe ratio over all candidate trajectories, not `1 - collision`.
- `pred_cost`: mean teacher cost of the model-selected final trajectory.

Do not confuse:

```text
collision = selected trajectory collision rate
safe = all candidate safe rate
```

They are not complements.

## Checkpoint Policy

`train.py` now saves best checkpoint using safety first:

```text
lower pred_collision_rate wins
if collision rate ties, lower traj_l2_error wins
```

Current best checkpoint path:

```text
outputs/checkpoints/best_model.pt
```

## Recent Training Results

After adding online teacher candidate targets but before collision margin, a
200-epoch run reached roughly:

```text
traj_l2 ~0.737
traj_pair ~0.637
traj_vel ~0.922
model_topk ~0.77
```

But it still selected collision trajectories in some custom scenarios.

After adding collision margin and using `NUM_SAMPLES = 1024`, a run was stopped
by Ctrl-C around epoch 76. Best checkpoint was saved at epoch 75:

```text
epoch 075
loss 5.0453
path_loss 3.5614
trajectory_loss 1.4837
collision_margin_loss 0.000315
model_topk 0.747
traj_vel 0.927
traj_pair 0.669
traj_l2 0.765
end_l2 0.992
pred_collision_rate 0.0625
candidate_safe_rate 0.8625
pred_teacher_cost 62611.0
```

This checkpoint was saved correctly before the Ctrl-C. `test_random.py` loads
epoch 75.

## Known Behavior From Custom Tests

Two important custom cases:

### Good case

```text
reference: p52 v151
prediction: p111 v151
pred_collides: False
pred_teacher_cost: 97.4386
reference_cost: 104.1774
```

Prediction is outside teacher top candidate set, but has lower teacher cost and
does not collide. This is good. It shows model can overcome teacher top-8 path
pruning.

### Bad case

```text
reference: p222 v151
prediction: p52 v151
pred_collides: True
pred_teacher_cost: ~1e6
```

The model still selects a collision path in this hard case. Collision margin
reduced average collision rate, but did not solve all hard collision-boundary
cases.

Important interpretation:

The same path index can be safe in one obstacle layout and colliding in another.
Do not judge a path index globally. Safety is scene-dependent.

## Current Open Problems

1. Collision rate is improved but not zero. Current best is about 6.25%.
2. Some hard cases still choose a colliding candidate with high model trajectory
   score.
3. Dataset obstacle sampling is globally random, so hard route-adjacent collision
   cases may be underrepresented.
4. `model_topk` around 0.74 means the path branch still misses teacher best paths
   about 25% of the time, but the more urgent issue is collision ranking.
5. Best checkpoint is safety-first now, but no separate validation set exists.
   Current metrics are training-set metrics.

## Recommended Next Steps

1. Test current epoch-75 checkpoint on several fixed `test_random.py` custom cases.
2. If continuing training, resume or retrain to 150-200 epochs and watch:

```text
collision
pred_cost
traj_l2
traj_pair
model_topk
```

3. If collision plateaus around 0.05-0.07:

```python
collision_margin_weight = 1.0
```

4. Route-aware hard obstacle sampling has been added. If collision still
   plateaus, tune its probability/ranges or add a more structured lane-crossing
   dynamic obstacle sampler.

5. Add an evaluation script that runs many fixed random/custom cases and reports:

```text
collision_rate
mean pred_teacher_cost
mean cost_gap = pred_cost - reference_cost
better_than_reference_rate
```

Do not rely only on visual inspection of one or two cases.

## Common Commands

Train:

```bash
cd /home/yihang/Documents/Alg_study/TorchStudy/ToySparseDriveV2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train.py
```

Note: PyTorch warns that `PYTORCH_CUDA_ALLOC_CONF` is deprecated and suggests
`PYTORCH_ALLOC_CONF`, but the old env var still worked in local runs.

Test custom/random case:

```bash
cd /home/yihang/Documents/Alg_study/TorchStudy/ToySparseDriveV2
python test_random.py
```

Teacher debug visualization:

```bash
cd /home/yihang/Documents/Alg_study/TorchStudy/ToySparseDriveV2/dataset
python teacher.py
```

Dataset visualization:

```bash
cd /home/yihang/Documents/Alg_study/TorchStudy/ToySparseDriveV2/dataset
python visualize_dataset.py
```

## Files Not To Touch Casually

- Do not modify real SparseDriveV2 vocab files under `SparseDriveV2/ckpt/kmeans`.
- Do not change coordinate convention without checking `grid.py`, `teacher.py`,
  `model.py`, and all visualizers together.
- Do not remove teacher online scoring from `train.py`; it is the main fix that
  made trajectory ranking meaningful.
- Do not treat `vel_acc` as the main failure signal while velocity loss weight is
  zero.

## If A New Codex Session Starts

Read this file first, then inspect:

```text
train.py
losses/losses.py
dataset/teacher.py
test_random.py
```

The latest major change was adding explicit `collision_margin_loss` and safety
first checkpoint selection. The latest known checkpoint is epoch 75 with about
6.25% selected-trajectory collision rate on training metrics.

## 2026-06-23 Update: Explicit Collision Head

ToySparseDriveV2 now follows SparseDriveV2 more closely by keeping trajectory
candidates and adding an explicit candidate-level safety head instead of relying
only on one scalar `trajectory_scores` output.

Changed files:

- `models/model.py`
  - Added `no_collision_head` beside `trajectory_score_head`.
  - `score_trajectories()` now returns `no_collision_logits` with shape `[B, C]`.

- `losses/losses.py`
  - `compute_model_candidate_losses()` now accepts `no_collision_logits`.
  - Added BCE loss against `~candidate_collision` as `no_collision_loss`.
  - `collision_margin_loss` now uses combined planning scores:
    `planning_scores = trajectory_scores + safety_score_weight * no_collision_logits`.

- `train.py`
  - Added `SAFETY_SCORE_WEIGHT = 2.0`.
  - Final candidate selection and metrics now use `planning_scores`, not raw
    `trajectory_scores`.
  - Logs new metrics: `no_col` and `no_col_acc`.

- `test_random.py`
  - Uses the same `planning_scores` for prediction.
  - Prints raw trajectory score, no-collision logit, and combined planning score
    for reference and prediction candidates.

Important: old checkpoints do not contain `no_collision_head` parameters. Retrain
after this architecture change before running meaningful `test_random.py` results.

## 2026-06-23 Update: Safety Weight Reduced + Raw Metrics

After observing that `SAFETY_SCORE_WEIGHT = 2.0` made planning very safe but
kept `traj_l2` high (~9-10m after 100+ epochs), the weight was reduced:

```python
SAFETY_SCORE_WEIGHT = 1.0
```

This was changed in both `train.py` and `test_random.py`.

`train.py` now logs raw trajectory-score argmax metrics alongside planning-score
argmax metrics:

- `raw_col`: collision rate from raw `trajectory_scores.argmax()`
- `raw_l2`: trajectory L2 from raw `trajectory_scores.argmax()`
- `raw_pair`: pair accuracy from raw `trajectory_scores.argmax()`

Existing `collision`, `traj_l2`, and `traj_pair` are still computed from:

```python
planning_scores = trajectory_scores + SAFETY_SCORE_WEIGHT * no_collision_logits
```

Use these paired metrics to diagnose whether the safety head is improving final
selection or overpowering the geometric/teacher trajectory score.

## 2026-06-24 Update: Teacher Dynamics Feasibility

`dataset/teacher.py` now checks velocity-vocab compatibility with the current ego speed.

New `TeacherConfig` fields:

```python
max_accel = 3.0
max_decel = 5.0
dynamics_invalid_weight = 1.0e5
accel_violation_weight = 20.0
```

Both the transition from current ego speed to the first future velocity and all
subsequent velocity steps are checked. Invalid profiles receive a large penalty,
while continuous acceleration violation and the existing acceleration/jerk costs
remain available for ranking and debugging. Teacher top-k logs now print
`dyn_invalid`, `init_violation`, and `accel_violation`.

Validation with ego speed 1 m/s selected v207 with initial acceleration about
1.996 m/s^2 and `dyn_invalid=False`, replacing the previously selected v151
profile that jumped to about 9.9 m/s at t=0.5 s.

## 2026-06-24 Update: Route Cost Uses Frenet Lateral Offset

Teacher route cost no longer uses nearest distance to discrete route points.
Each path/trajectory point is projected exactly onto the nearest route polyline
segment, producing signed Frenet lateral offset `l` (`l>0` left, `l<0` right).

Trajectory route cost is the weighted mean of `abs(l)` over valid trajectory
points. Weights increase linearly from `0.5` to `1.5`, so later points matter
more and trajectories are encouraged to return toward the route after avoidance.
Path coarse filtering now uses the same segment-projection definition with an
unweighted mean `abs(l)`. `route_l` is included in teacher debug output.

Exact segment projection is used instead of route densification because it gives
a continuous projection independent of route point spacing.



## Full-Search Teacher Cache (2026-06-24)

The old dataset Teacher used path coarse filtering before combining paths with
velocity vocab. That could remove a route-following path which would be safe
when paired with a braking velocity. This old behavior is now only a debugging
fallback.

The authoritative training labels now come from an offline full search:

```text
1024 paths x 256 velocities = 262,144 trajectories per scene
```

Implementation:

- `dataset/teacher.py::score_all_trajectories_chunked`
  scores every path/velocity pair in path chunks and only retains the global
  Top-K plus the best cost for each of the 1024 paths.
- `dataset/build_teacher_cache.py`
  generates resumable per-sample `.npz` caches.
- `dataset/dataset.py`
  deterministically regenerates the scene and loads the corresponding cached
  labels. It validates cache version, sample index, seed offset, and route path.
- `train.py`
  requires `cache/teacher_v1`; it does not silently fall back to the old
  coarse-filter Teacher.

Generate the complete cache before training:

```bash
cd TorchStudy/ToySparseDriveV2/dataset
conda activate torch310
python build_teacher_cache.py \
  --num-samples 1024 \
  --path-chunk-size 32 \
  --top-k 64 \
  --output-dir ../cache/teacher_v1
```

The command is resumable: existing sample files are skipped. Use `--overwrite`
only when Teacher scoring logic or weights change.

A smoke run on this machine took about 15.6 seconds for one scene with chunk
size 32, so a sequential 1024-scene build is expected to take roughly 4-5
hours. The cached files are then reused for every epoch.

Training still performs online Teacher scoring only for the small mixed
candidate set produced by model Top-K paths plus cached Teacher paths. This
keeps explicit collision/cost supervision on current model mistakes without
repeating the full 262,144-trajectory search.

Smoke validation result for sample 0:

```text
old coarse Teacher: p372 / v137
full-search Teacher: p4 / v137
```

This confirms that the old path pruning could change the generated truth.


## Teacher Safety/Comfort Cost Update (2026-06-24)

Teacher trajectory scoring now uses:

```text
minimum_clearance = 0.8 m
clearance_violation_weight = 200
route_weight = 1.5
progress_weight = 0.1
lateral_accel_weight = 0.1
```

`clearance_violation = max(0.8 - clearance, 0)^2` prevents trajectories with
only a few centimeters of residual clearance from winning merely because they
do not geometrically overlap an obstacle. Lateral acceleration is estimated as
`speed * yaw_rate` and penalized by its masked mean square.

On the 8 m/s manual Teacher scene, the old winner `p179/v68` had only 0.047 m
clearance. After this update the winner is `p552/v114`, which stays on the route,
has 2.387 m clearance, and decelerates from 8.0 m/s to about 0.9 m/s over 4 s.

Teacher cache version is now 2. Any cache generated before this update must be
regenerated.


## Scene Generation and Cache v3 (2026-06-25)

Training scene generation now addresses three data-quality problems:

1. Each scene samples ego speed uniformly from `[0, 10] m/s`.
2. Obstacles whose initial expanded AABB overlaps the ego vehicle are rejected
   and resampled.
3. During full-search cache generation, a scene is rejected and deterministically
   regenerated when its retained global candidates contain no collision-free
   trajectory. The accepted `scene_attempt` is stored in the cache so Dataset
   reconstruction is exact.

`ToySparseDriveV2Dataset.generate_scene(index, scene_attempt)` now returns:

```text
route_path_index, route_path, goal_xy, ego_state, obstacles
```

The random generator uses `SeedSequence([seed_offset, index, scene_attempt])`.
Cache version is now 3 and stores `scene_attempt` plus `ego_speed`. `train.py`
preflights all 1024 files and rejects missing, corrupt, or non-v3 caches.

The previous v2 cache is incompatible and must be regenerated:

```bash
cd TorchStudy/ToySparseDriveV2
python -u dataset/build_teacher_cache.py \
  --num-samples 1024 \
  --path-chunk-size 32 \
  --top-k 64 \
  --output-dir cache/teacher_v1 \
  --overwrite
```

A 3-scene smoke test produced ego speeds 2.70, 5.57, and 4.02 m/s, no initial
obstacle overlap, and collision-free Teacher best trajectories.
