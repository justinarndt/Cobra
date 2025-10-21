// src/jit/GaudiDialect.td
//
// This TableGen file defines a new, highly specialized "GaudiGraph" dialect.
// The operations in this dialect do not represent computation directly.
// Instead, they are a 1:1 mapping to the C++ API calls required to build
// a computation graph using Intel's SynapseAI SDK. This abstraction allows
// us to represent the target program as an MLIR graph.

#ifndef GAUDI_DIALECT
#define GAUDI_DIALECT

include "mlir/IR/DialectBase.td"

def Gaudi_Dialect : Dialect {
 let name = "gaudi";
 let summary = "A dialect for representing SynapseAI computation graphs.";
 let cppNamespace = "::cobra::mlir";
}

#endif // GAUDI_DIALECT