// src/jit/lowering/LinalgToGPU.cpp
//
// This pass converts operations from the `linalg` dialect to the `gpu`
// dialect. This is a critical step in targeting GPUs. It leverages MLIR's
// standard dialect conversion infrastructure to tile the loops within
// linalg ops for parallelism and then maps those parallel loops onto the
// grid/block/thread hierarchy of a GPU, creating a `gpu.launch_func` op.

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

// This pass will be composed of standard MLIR passes.
// In the main compiler pipeline, we will create a PassManager
// and add the following passes in sequence:
// 1. A pass to tile `linalg.generic` operations to create `scf.parallel` loops.
// 2. A pass that converts those `scf.parallel` loops into `gpu.launch_func` operations.
// This modular approach allows us to reuse the robust, well-tested components
// provided by the MLIR project for this complex transformation.