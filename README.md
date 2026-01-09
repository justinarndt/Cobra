# Cobra 🐍⚡

[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LLVM](https://img.shields.io/badge/LLVM-15+-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Write Idiomatic Python. Run at Native Speed. Deploy Anywhere.**

> ⚠️ **ACTIVE DEVELOPMENT:** Cobra is a research project in early stages. Core features are being actively developed. Not production-ready.

## The Vision

Python's expressiveness comes at a cost—interpreted execution is slow. Existing acceleration tools force you to choose:

- **Numba/JAX:** Fast, but restrictive syntax and limited hardware support
- **PyTorch/TensorFlow:** Powerful, but heavyweight and domain-specific
- **Cython:** Requires rewriting Python in a hybrid language
- **PyPy:** JIT compilation, but limited ecosystem compatibility

**Cobra aims to be different:** A unified JIT runtime that compiles idiomatic Python to native machine code with near-zero overhead, targeting any hardware (NVIDIA GPUs, AMD GPUs, CPUs) through a single codebase.

## Architecture
```
Python Source Code
       ↓
  AST Parser → Type Inference
       ↓
  LLVM IR Generation
       ↓
  ┌─────────────────────────┐
  │   Target-Specific       │
  │   Code Generation       │
  ├─────────┬─────────┬─────┤
  │ NVIDIA  │   AMD   │ CPU │
  │  CUDA   │  ROCm   │ x86 │
  └─────────┴─────────┴─────┘
       ↓
  Native Execution
```

**Core Components:**

- **Custom Memory Manager (C++):** RAII-based arena allocator with GPU-aware memory pooling
- **LLVM-based JIT:** Compiles Python AST → LLVM IR → native machine code
- **Unified Backend:** Single abstraction layer for CUDA, ROCm, and CPU execution
- **Type Inference Engine:** Analyzes Python code to generate optimal static types

## Current Status

| Feature                     | Status        |
|-----------------------------|---------------|
| Basic JIT Compilation       | 🟡 In Progress |
| Memory Manager (CPU)        | 🟢 Working     |
| GPU Memory Pooling          | 🟡 In Progress |
| CUDA Backend                | 🔴 Planned     |
| ROCm Backend                | 🔴 Planned     |
| Type Inference              | 🟡 In Progress |
| NumPy Integration           | 🔴 Planned     |
| Multi-threading Support     | 🔴 Planned     |

## Installation (Development Build)

**Prerequisites:**
- LLVM 15+
- CMake 3.20+
- Python 3.8+
- C++17 compatible compiler
```bash
git clone https://github.com/justinarndt/Cobra.git
cd Cobra
git submodule update --init --recursive

# Build C++ components
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install Python package (editable)
cd ..
pip install -e .
```

## Usage Example (Planned API)
```python
from cobra import jit

@jit
def matrix_multiply(A, B):
    """Standard Python syntax - compiled to native code"""
    N, M = A.shape
    M, K = B.shape
    C = [[0.0] * K for _ in range(N)]
    
    for i in range(N):
        for j in range(K):
            for k in range(M):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# Automatically dispatches to best available hardware
result = matrix_multiply(A_gpu, B_gpu)  # Runs on GPU
result = matrix_multiply(A_cpu, B_cpu)  # Runs on CPU
```

## Benchmarks (Preliminary)

> 🔬 **Note:** Benchmarks are preliminary and not yet reproducible. Full benchmark suite in development.

| Operation         | Python  | Cobra (Target) | Speedup |
|-------------------|---------|----------------|---------|
| Matrix Multiply   | 1.0x    | ~50x           | 50x     |
| Element-wise Ops  | 1.0x    | ~100x          | 100x    |
| Reductions        | 1.0x    | ~80x           | 80x     |

## Roadmap

**Phase 1: Foundation (Current)**
- [x] Project structure and build system
- [x] Basic memory manager
- [ ] LLVM IR generation for core Python subset
- [ ] Type inference for common patterns

**Phase 2: GPU Support**
- [ ] CUDA kernel generation
- [ ] Memory transfer optimization
- [ ] GPU memory pooling
- [ ] ROCm backend

**Phase 3: Ecosystem Integration**
- [ ] NumPy compatibility layer
- [ ] PyTorch interop
- [ ] Automatic parallelization
- [ ] Profiling and debugging tools

**Phase 4: Production Readiness**
- [ ] Comprehensive test suite
- [ ] Performance benchmarking framework
- [ ] Documentation and tutorials
- [ ] Package distribution (PyPI)

## Why Cobra?

**Design Principles:**

1. **Zero Compromise on Syntax:** Write normal Python. No decorators everywhere, no special syntax
2. **Hardware Agnostic:** One codebase runs on NVIDIA, AMD, or CPU
3. **Transparent Performance:** Clear cost model for what gets accelerated
4. **Ecosystem Friendly:** Works with existing Python packages where possible

## Project Structure
```
/
├── cobra/           # Python frontend (AST parsing, type inference)
├── src/             # C++ core (JIT compiler, memory manager)
│   ├── compiler/    # LLVM IR generation
│   ├── runtime/     # Execution engine and memory management
│   └── backends/    # Target-specific code (CUDA, ROCm, CPU)
├── benchmarks/      # Performance comparison suite
├── examples/        # Usage examples and tutorials
└── tests/           # Unit and integration tests
```

## Contributing

Cobra is in early development. Contributions are welcome, but expect frequent breaking changes. If you're interested in:

- Compiler development (LLVM, type systems)
- GPU programming (CUDA, ROCm)
- Python runtime internals
- Performance optimization

Feel free to open an issue or reach out at [justinarndtai@gmail.com](mailto:justinarndtai@gmail.com).

## Comparison to Existing Tools

| Tool      | Syntax        | GPU Support | Multi-Backend | JIT Speed |
|-----------|---------------|-------------|---------------|-----------|
| NumPy     | Python-like   | No          | No            | N/A       |
| Numba     | Restricted    | NVIDIA only | No            | Fast      |
| JAX       | Restricted    | NVIDIA/TPU  | Partial       | Fast      |
| PyTorch   | Framework     | Yes         | Partial       | Medium    |
| Cobra     | **Pure Python** | **Yes**     | **Yes**       | **Fast**  |

## License

MIT

## Acknowledgments

Built with LLVM, inspired by PyPy, Numba, and decades of JIT compiler research.

---

**Status Updates:** Follow development progress in the [Issues](https://github.com/justinarndt/Cobra/issues) tab. Major milestones will be announced as releases.
