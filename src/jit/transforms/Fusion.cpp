// src/jit/transforms/Fusion.cpp
//
// This file implements one of Cobra's most important optimizations:
// kernel fusion. This MLIR pass operates on the `linalg` dialect. It
// searches for patterns where one `linalg.generic` operation produces a
// result that is immediately consumed by another `linalg.generic` op.
// When this pattern is found, the pass merges the two operations into a
// single, larger `linalg.generic` op. This is a profound performance
// optimization as it completely eliminates the need to write the
// intermediate result to memory, saving significant memory bandwidth.

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

// The implementation of this pass will use MLIR's powerful Pattern
// Rewriter infrastructure to find and replace the adjacent linalg ops.
// It will inspect the producers and consumers of tensor values within
// a block to identify fusion opportunities.
//
// For example, it will transform:
// %1 = linalg.generic ins(%A, %B) outs(%tmp) { ... } -> tensor<...>
// %2 = linalg.generic ins(%1, %C) outs(%D) { ... } -> tensor<...>
//
// Into:
// %2 = linalg.generic ins(%A, %B, %C) outs(%D) { /* merged body */ } -> tensor<...>