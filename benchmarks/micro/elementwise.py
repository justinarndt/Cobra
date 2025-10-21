# benchmarks/micro/elementwise.py
#
# Compares the performance of a fused element-wise operation like
# `d = (a * b) + c`. This benchmark is specifically designed to highlight the
# architectural advantage of Cobra's kernel fusion. Libraries like NumPy
# must create a large, temporary intermediate array for the result of `a * b`,
# consuming significant memory bandwidth. Cobra performs the entire
# operation in a single pass over the data.

import numpy as np
import cobra
from benchmarks.runner import run_benchmark

# Define the operation in a JIT-compiled function
@cobra.jit
def fused_op_cobra(df):
    df['d'] = (df['a'] * df['b']) + df['c']

def fused_op_numpy(a, b, c):
    return (a * b) + c

def main():
    print("--- Running Fused Element-wise Benchmark ---")
    size = 20_000_000
    
    # NumPy setup
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    c_np = np.random.rand(size).astype(np.float32)
    
    # Cobra setup
    frame_data = {'a': a_np, 'b': b_np, 'c': c_np}
    cf = cobra.CobraFrame(frame_data)

    # Run benchmarks
    cobra_time = run_benchmark(fused_op_cobra, (cf,))
    numpy_time = run_benchmark(fused_op_numpy, (a_np, b_np, c_np))

    print(f"NumPy (baseline): {numpy_time:.4f} ms")
    print(f"Cobra JIT:        {cobra_time:.4f} ms")
    print(f"Speedup: {numpy_time / cobra_time:.2f}x")

if __name__ == "__main__":
    main()