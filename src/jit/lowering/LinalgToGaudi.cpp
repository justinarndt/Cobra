// src/jit/lowering/LinalgToGaudi.cpp
//
// This file implements the conversion from the hardware-agnostic `linalg`
// dialect to our Gaudi-specific `GaudiGraph` dialect. This pass is the
// bridge between computation and graph representation. It contains rewrite
// patterns that match standard compute operations (like `linalg.add`) and
// replace them with the corresponding graph-building node (`gaudi.create_node`).

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "jit/GaudiOps.h" // Our custom Gaudi operations
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace cobra::mlir;

namespace {
// This pattern matches a `linalg.add` operation and converts it into
// a `gaudi.create_node` operation with the `op_type` attribute set to "add".
struct LinalgAddToGaudi : public OpConversionPattern<linalg::AddOp> {
    using OpConversionPattern<linalg::AddOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(linalg::AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<CreateNodeOp>(
            op,
            rewriter.getStringAttr("add"), // op_type
            adaptor.getInputs(),           // inputs
            op->getResultTypes()           // output types
        );
        return success();
    }
};

// Similar patterns would be created for linalg.matmul, linalg.mul, etc.

} // end anonymous namespace

// ... Pass registration logic ...