// src/jit/GaudiOps.td
//
// Defines the custom operations for the GaudiGraph dialect. The primary
// operation is `gaudi.create_node`, which directly corresponds to the
// `synNodeCreate` function in the SynapseAI API.

#ifndef GAUDI_OPS
#define GAUDI_OPS

include "GaudiDialect.td"
include "mlir/IR/OpBase.td"

// Base class for all Gaudi dialect operations.
class Gaudi_Op<string mnemonic, list<Trait> traits = []> :
   Op<Gaudi_Dialect, mnemonic, traits>;

// Represents a call to `synNodeCreate`. It takes the operation type
// (e.g., "add", "matmul") as a string attribute and the input tensors
// as arguments. It returns the resulting output tensor.
def CreateNodeOp : Gaudi_Op<"create_node"> {
 let summary = "Represents a call to the synNodeCreate API function.";
 let arguments = (ins
    StringAttr:$op_type,
    Variadic<AnyTensor>:$inputs
 );
 let results = (outs AnyTensor:$output);
}

#endif // GAUDI_OPS