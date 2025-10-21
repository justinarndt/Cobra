// src/jit/emitter/GaudiEmitter.cpp
//
// This file contains a special kind of compiler pass: an emitter.
// Instead of transforming MLIR into more MLIR, this pass iterates over the
// `GaudiGraph` dialect operations and prints out a string of C++ source code.
// The generated code is a complete, compilable program that uses the
// SynapseAI SDK to build and run the computation graph that was represented
// in MLIR. This is the final step in the Gaudi compiler backend.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "jit/GaudiOps.h"
#include <string>
#include <sstream>

using namespace mlir;
using namespace cobra::mlir;

// Function to walk the MLIR module and generate the C++ source string.
std::string emitCppForGaudi(ModuleOp module) {
    std::stringstream ss;

    // Preamble: includes and main function structure.
    ss << "#include <synapse_api.h>\n";
    ss << "extern \"C\" void execute_graph(void** args) {\n";
    ss << "  synGraphHandle graph;\n";
    ss << "  synGraphCreate(&graph, 0);\n";

    // Walk every operation in the module's main function.
    module.walk([&](CreateNodeOp op) {
        // For each `gaudi.create_node`, print a line of C++
        // that calls the corresponding SynapseAI API function.
        ss << "  synNodeCreate(graph, \"" << op.getOpType().str() << "\", ...);\n";
    });

    // Postamble: compile and execute the graph.
    ss << "  synGraphCompile(graph);\n";
    ss << "  synGraphRun(graph, ...);\n";
    ss << "}\n";

    return ss.str();
}```

##### Step 2.4.4.1: Integrate Gaudi Runtime
*   **Git Commit:** `feat(runtime): add support for compiling and running SynapseAI graphs`
*   **File:** `src/memory/MemoryManager.h`

```cpp
// MODIFIED: src/memory/MemoryManager.h
// ... other includes ...
#include <string>

class MemoryManager {
public:
    // ... existing public methods ...

    // ADDED: A new method to handle the Gaudi execution flow.
    void executeGaudiGraph(const std::string& cpp_source);

private:
    // ... existing private members ...
};