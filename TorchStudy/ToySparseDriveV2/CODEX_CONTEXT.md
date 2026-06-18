# ToySparseDriveV2 Codex Context

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

Current obstacle sampling is globally random:

```text
num obstacles: 4 to 12
x: grid_config.x_min + 8 to grid_config.x_max - 8
y: grid_config.y_min + 8 to grid_config.y_max - 8
length: 3 to 8
width: 2 to 5
vx: -4 to 4
vy: -3 to 3
```

This is enough for basic training but not ideal for hard collision-boundary
learning. Future improvement: add route-aware obstacle sampling near the route,
while keeping some global random obstacles.

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

4. Add a route-aware hard obstacle sampler:

```text
70% obstacles near route / future ego area
30% global random obstacles
```

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
