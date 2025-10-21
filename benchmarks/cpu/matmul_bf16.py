# benchmarks/cpu/matmul_bf16.py
#
// Benchmarks a large bfloat16 matrix multiplication on a supported
# Intel Xeon CPU (e.g., Sapphire Rapids). This benchmark is critical for
# proving the value of the specialized AMX code generation path. It will
# compare the performance of three JIT compilation strategies:
# 1. A naive loop-based implementation.
# 2. A version JIT-compiled with AVX-512 vectorization.
# 3. The version JIT-compiled with the AMX lowering pass enabled.
# The expected result is a dramatic, order-of-magnitude speedup for the
# AMX version, showcasing the power of compiling directly to specialized hardware.

import cobra
import numpy as np

# Assuming bfloat16 is a type Cobra understands
# For real use, this would require a custom dtype extension.
bf16 = np.dtype("bfloat16")

@cobra.jit(target='cpu', flags=['-enable-amx'])
def matmul_amx(a, b):
    return a @ b

@cobra.jit(target='cpu')
def matmul_avx512(a, b):
    return a @ b

# ... Benchmark runner code ...
# 1. Create two large (e.g., 2048x2048) matrices with bfloat16 data.
# 2. Convert them to cobra.array objects.
# 3. Time the execution of matmul_amx.
# 4. Time the execution of matmul_avx512.
# 5. Print a comparison of the results, calculating the speedup factor.