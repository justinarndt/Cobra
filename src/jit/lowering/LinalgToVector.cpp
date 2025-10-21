// src/jit/lowering/LinalgToVector.cpp
//
// This pass begins the CPU-specific code generation path. It converts
// `linalg` operations into operations on the `vector` dialect. The
// `vector` dialect is a hardware-agnostic way to express SIMD (Single
// Instruction, Multiple Data) computations. By lowering to this dialect,
// we can perform vector-level optimizations before generating final
// machine-specific LLVM IR.

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Pass/Pass.h"

// The implementation of this stage will be handled by a standard MLIR
// conversion pass. In our main compiler pipeline, we will add the
// `mlir::createLinalgGeneralizationPass()` and
// `mlir::createConvertLinalgToLoopsPass()` to transform `linalg` ops
// into standard loops, which can then be vectorized.