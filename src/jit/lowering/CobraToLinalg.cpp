// src/jit/lowering/CobraToLinalg.cpp
//
// This file implements the first stage of lowering. It converts operations
// from our high-level, Python-semantic Cobra dialect into the `linalg`
// dialect. The `linalg` dialect is a powerful, structured way to represent
// loop nests and tensor operations, making it the perfect target for
// optimizations like fusion before we generate hardware-specific code.

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "jit/CobraOps.h" // Our custom operations
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace cobra::mlir;

namespace {
// This pattern converts a `cobra.fused_kernel` operation into a
// `linalg.generic` operation. The body of the `fused_kernel` is
// moved directly into the body of the `linalg.generic` op.
struct FusedKernelLowering : public OpConversionPattern<FusedKernelOp> {
    using OpConversionPattern<FusedKernelOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(FusedKernelOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // Create a linalg.generic op to replace the cobra kernel.
        auto linalgOp = rewriter.create<linalg::GenericOp>(
            op.getLoc(),
            /*resultTensorTypes=*/op->getResultTypes(),
            /*inputs=*/adaptor.getInputs(),
            /*outputs=*/adaptor.getOutputs(),
            /*indexingMaps=*/...); // Indexing maps would be created here

        // Move the body of the cobra op into the new linalg op.
        rewriter.inlineRegionBefore(op.getBody(), linalgOp.getRegion(), linalgOp.getRegion().begin());

        rewriter.replaceOp(op, linalgOp.getResults());
        return success();
    }
};
} // end anonymous namespace

// Function to register all the patterns for this pass.
void populateCobraToLinalgConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<FusedKernelLowering>(patterns.getContext());
}