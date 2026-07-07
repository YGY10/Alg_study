# AGENTS.md

Guidelines for coding agents working in `/home/yihang/Documents/Alg_study`.

## Repository Shape

This is an algorithm study workspace with small standalone programs and several larger
autonomous-driving experiments. Expect mixed C++ and Python.

- `Basics/`, `LeetCode/`, `LeetCodeTest/`: data structures, language examples, and
  coding-practice solutions.
- `A_Star/`, `Dijkstra/`, `Basic_Geometry/`, `Bayesian_Filter/`, `Clustering/`,
  `State_Machine/`: focused algorithm implementations, usually runnable one file at a
  time.
- `Simulation/`, `Prediction/`, `AvoidCrubRL/`, `CarlaProject0910/`,
  `CarlaProject0916/`: planning, simulation, reinforcement-learning, and CARLA-related
  experiments.
- `TorchStudy/`, `Theory/`, `SparseDriveV2study/`: PyTorch, transformer, and model-study
  scripts.
- `GoalFlow/`, `SparseDriveV2/`: package-style Python projects with their own
  `requirements.txt`, `setup.py`, and README files.
- `eigen-3.4.0/`, `libtorch/`: vendored or local third-party dependencies. Do not edit
  them unless the task explicitly targets those directories.
- `include/`: shared headers.
- `output/`, `build/`: generated build artifacts.

## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a
tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are
sub-millisecond and return structural information grep cannot.

Prefer CodeGraph for structural questions: where a symbol is defined, what calls it, what
it calls, how a flow reaches another function, what would break after a signature change,
or when broad context is needed before editing. Use native `rg`, `sed`, or file reads for
literal text queries, generated files, configs, and exact command output.

Useful patterns:

- Start with `codegraph_explore` for most code-understanding or edit tasks; it returns
  relevant symbol source grouped by file.
- Use `codegraph_trace` for "how does X reach Y" instead of rebuilding the path with
  grep and callers.
- If CodeGraph reports pending sync or stale files, read only those listed files from
  disk before editing.
- If `.codegraph/` is not initialized, ask before running `codegraph init -i`.

## Build And Run

There is no single root build system. Prefer the smallest command that validates the file
or subproject you changed.

### Simple C++

```bash
mkdir -p output
g++ -std=c++17 -g path/to/file.cpp -I. -I./eigen-3.4.0 -o output/file.out
./output/file.out
```

### C++ With Local LibTorch/CUDA/Python Linkage

The VSCode task in `.vscode/tasks.json` links against `libtorch`, CUDA, and Python 3.8.
Use it as the reference when a C++ file includes Torch, CUDA, Python, or
`matplotlibcpp.h`.

```bash
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -g path/to/file.cpp \
    -I. -I./eigen-3.4.0 -I./eigen-3.4.0/Eigen \
    -I/usr/include/python3.8 \
    -I./libtorch/include -I./libtorch/include/torch/csrc/api/include \
    -I/usr/local/cuda/include \
    -L./libtorch/lib -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu \
    -Wl,-rpath,./libtorch/lib -Wl,-rpath,/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
    -lpython3.8 -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcuda -lcudart \
    -o output/file.out
```

### Python

```bash
python3 path/to/script.py
CUDA_VISIBLE_DEVICES=0 python3 path/to/script.py
```

For `GoalFlow/` or `SparseDriveV2/`, read the local README first and run commands from
that subdirectory when possible.

## Formatting

- C++ uses `.clang-format` at the repo root: 4-space indentation, 100-column limit,
  attached braces, sorted includes, C++17-compatible style.
- Python should follow Black formatting. Use 4 spaces and keep imports grouped as
  standard library, third-party, then local.

Commands:

```bash
clang-format -i path/to/file.cpp
clang-format --dry-run --Werror path/to/file.cpp
black path/to/file.py
black --check path/to/file.py
```

Only format files you intentionally touched unless the user asks for broader cleanup.

## Testing And Verification

No formal root test suite is configured. Validate narrowly:

- For C++ changes, compile the affected `.cpp` and run the resulting binary from
  `output/`.
- For Python changes, run the edited script or the nearest small script that imports the
  changed code.
- For package-style projects, prefer their local README or setup instructions.
- If dependencies such as CARLA, CUDA, PyTorch, or NAVSIM are unavailable, still run
  static checks where practical and clearly report what could not be executed.

Known examples:

```bash
python3 Bayesian_Filter/Kalman_Filter.py
python3 Simulation/simulation_main.py
./output/Filter_test.out
```

## Coding Guidelines

- Keep changes scoped to the requested algorithm, experiment, or subproject.
- Preserve existing naming and file organization in the touched directory.
- C++: use C++17, RAII, `std::vector` for dynamic arrays, Eigen for matrix math, and
  `torch::Tensor` only where the surrounding code already uses LibTorch.
- Python: use type hints when they clarify function contracts; use NumPy/PyTorch
  vectorization for numerical code when it stays readable.
- Prefer assertions for small study scripts; use explicit exceptions for reusable helpers
  or invalid public inputs.
- Do not modify vendored dependencies, model weights, datasets, logs, or generated
  artifacts unless the task requires it.

## Documentation

- Chinese comments and notes are common in this repository; keep that style when adding
  user-facing explanations near existing Chinese documentation.
- Use English for concise technical comments when that matches the surrounding code.
- For new algorithm notes, include the core idea, complexity, and a minimal runnable
  example.

## Git And Safety

- The worktree may contain user changes. Do not revert or overwrite unrelated edits.
- Avoid destructive commands such as `rm`, `git reset`, or branch checkout unless the
  user explicitly requests them or approves the action.
- Keep generated outputs in `output/` or a task-specific temporary path.
