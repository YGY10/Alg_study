# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this algorithm study repository.

## Repository Overview

This is an algorithm study repository focused on autonomous driving, path planning, simulation, and machine learning. The codebase uses mixed C++/Python with PyTorch integration for trajectory planning and neural network models.

## Build Commands

### C++ Build
```bash
# Primary build command (configured in .vscode/tasks.json)
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -g ${file} \
    -I${workspaceFolder} \
    -I${workspaceFolder}/eigen-3.4.0 \
    -I${workspaceFolder}/eigen-3.4.0/Eigen \
    -I/usr/include/python3.8 \
    -I${workspaceFolder}/libtorch/include \
    -I${workspaceFolder}/libtorch/include/torch/csrc/api/include \
    -I/usr/local/cuda/include \
    -L${workspaceFolder}/libtorch/lib \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -Wl,-rpath,${workspaceFolder}/libtorch/lib \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
    -lpython3.8 -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcuda -lcudart \
    -o output/${fileBasenameNoExtension}.out

# Simple C++ build (for basic algorithms without external dependencies)
g++ -std=c++17 -g ${file} -I${workspaceFolder} -o output/${fileBasenameNoExtension}.out
```

### Python Execution
```bash
# Run Python files
python3 ${file}

# Run with specific GPU (if available)
CUDA_VISIBLE_DEVICES=0 python3 ${file}
```

## Lint and Format Commands

### C++ Formatting
```bash
# Format C++ files using clang-format
clang-format -i ${file}

# Check formatting without modifying
clang-format --dry-run --Werror ${file}
```

### Python Formatting
```bash
# Format Python files using Black
black ${file}

# Check formatting without modifying
black --check ${file}
```

## Test Commands

### Running Single Tests
```bash
# C++ test files
./output/Filter_test.out

# Python test files
python3 ./Torch/test_transformer_encoder.py
python3 ./Bayesian_Filter/Kalman_Filter.py
```

### Manual Testing
```bash
# Build and run in one command
g++ -std=c++17 -g ${file} -I${workspaceFolder} -o output/test.out && ./output/test.out
```

## Code Style Guidelines

### C++ Style (Google-based via clang-format)
- **Indentation**: 4 spaces
- **Line limit**: 100 characters
- **Pointer alignment**: Left (`int* ptr` not `int *ptr`)
- **Brace style**: Attach (`if (condition) {` not new line)
- **Standard**: C++17
- **Include order**: System headers, then external libraries, then local headers
- **Naming conventions**:
  - Classes: PascalCase (`class MyClass`)
  - Functions: snake_case (`void my_function()`)
  - Variables: snake_case (`int my_variable`)
  - Constants: UPPER_SNAKE_CASE (`const int MAX_SIZE`)

### Python Style (Black formatter)
- **Indentation**: 4 spaces
- **Line limit**: 88 characters (Black default)
- **String quotes**: Double quotes preferred
- **Naming conventions**:
  - Classes: PascalCase (`class MyClass:`)
  - Functions: snake_case (`def my_function():`)
  - Variables: snake_case (`my_variable = 1`)
  - Constants: UPPER_SNAKE_CASE (`MAX_SIZE = 100`)

### Import Guidelines

#### C++ Includes
```cpp
// System headers (alphabetical)
#include <algorithm>
#include <iostream>
#include <vector>

// External libraries (Eigen, libtorch)
#include <Eigen/Dense>
#include <torch/torch.h>

// Local headers (relative to workspace)
#include "my_header.h"
```

#### Python Imports
```python
# Standard library (alphabetical)
import math
import os
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import torch
from matplotlib import pyplot as plt

# Local imports
from .common import Vec2D
from .planner import TrajectoryPlanner
```

## Error Handling

### C++ Error Handling
- Use exceptions for error conditions (`std::runtime_error`, `std::invalid_argument`)
- Use assertions for debugging (`assert(condition)`)
- Return error codes for performance-critical code
- Use RAII for resource management

### Python Error Handling
- Use exceptions for error conditions (`ValueError`, `RuntimeError`)
- Use assertions for debugging (`assert condition, "error message"`)
- Use context managers for resource handling (`with open(...)`)
- Log errors rather than print for production code

## Type Guidelines

### C++ Types
- Use `auto` for type deduction when clear
- Prefer `int` over `int32_t` unless specific size needed
- Use `std::vector` for dynamic arrays
- Use `Eigen::VectorXd` for mathematical vectors
- Use `torch::Tensor` for neural network tensors

### Python Types
- Use type hints for function signatures and variables
- Use `List[T]` and `Dict[K, V]` from typing module
- Use `np.ndarray` for numerical arrays
- Use `torch.Tensor` for neural network tensors
- Use dataclasses for structured data

## Project Structure Guidelines

### Directory Organization
- `Basics/`: Fundamental algorithms and data structures
- `Basic_Geometry/`: Geometric calculations and utilities
- `Simulation/`: Main simulation framework and tools
- `Torch/`: PyTorch models and machine learning experiments
- `A_Star/`, `Dijkstra/`: Path planning algorithm implementations
- `State_Machine/`: State machine implementations
- `Clustering/`: Clustering algorithms
- `Bayesian_Filter/`: Filtering algorithms (Kalman, Particle)
- `include/`: Shared header files
- `output/`: Build output directory

### File Naming
- C++ files: PascalCase for classes (`MyClass.cpp`), snake_case for utilities (`math_utils.cpp`)
- Python files: snake_case (`trajectory_planner.py`)
- Header files: snake_case (`my_header.h`)

## Dependencies

### Required Libraries
- **Eigen 3.4.0**: Linear algebra (included in repository)
- **LibTorch 2.4.1**: PyTorch C++ API with CUDA support
- **Python 3.8**: Python integration
- **CUDA**: GPU acceleration (if available)

### Build Requirements
- **g++** with C++17 support
- **Python 3.8** development headers
- **CMake** (for Eigen library configuration)

## Development Environment

### IDE Configuration
- **VSCode** as primary IDE
- **C++/Python extensions** configured
- **Format on save** enabled for both languages
- **Integrated debugging** for C++ and Python
- **Custom build tasks** configured in `.vscode/tasks.json`

### Git Configuration
- Repository uses UTF-8 encoding
- Chinese comments and documentation are present
- Remote repository configured for collaboration

## Testing Guidelines

### C++ Testing
- No formal unit testing framework configured
- Use simple `assert()` statements for verification
- Create test functions with descriptive names
- Build test files to `output/` directory

### Python Testing
- No formal testing framework configured
- Use `assert` statements for simple verification
- Create test functions and call them in `if __name__ == "__main__":`
- Use descriptive test function names

## Performance Considerations

### C++ Performance
- Use `-O2` or `-O3` optimization for release builds
- Profile with `-g` and debugging tools
- Consider memory allocation patterns
- Use Eigen's optimized linear algebra operations

### Python Performance
- Use NumPy vectorization instead of loops
- Consider PyTorch tensors for GPU acceleration
- Profile with `cProfile` or `line_profiler`
- Use `@njit` decorator from Numba for critical sections

## Documentation Guidelines

### Code Comments
- Use Chinese comments for user-facing explanations (as per existing codebase)
- Use English comments for technical implementation details
- Include function parameter descriptions
- Add usage examples for complex algorithms

### File Documentation
- Create `.md` files for algorithm explanations
- Include mathematical formulations where relevant
- Add complexity analysis (time/space)
- Provide usage examples and test cases