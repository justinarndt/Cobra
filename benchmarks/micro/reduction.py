# benchmarks/micro/reduction.py
#
# Benchmarks the performance of a parallel reduction, such as `sum()`,
# on a large array across different backends and libraries. This tests
# the efficiency of the compiler in handling cross-thread communication
# and aggregation.

import numpy as np
import cobra
from benchmarks.runner import run_benchmark

@cobra.jit
def sum_cobra(arr):
    # A full implementation would need a `sum()` method on CobraArray
    # return arr.sum()
    pass # Placeholder

def sum_numpy(arr):
    return arr.sum()

def main():
    print("--- Running Reduction (Sum) Benchmark ---")
    size = 50_000_000
    
    arr_np = np.random.rand(size).astype(np.float32)
    arr_cobra = cobra.CobraArray(arr_np)

    # Run benchmarks
    # cobra_time = run_benchmark(sum_cobra, (arr_cobra,))
    numpy_time = run_benchmark(sum_numpy, (arr_np,))
    
    print(f"NumPy (baseline): {numpy_time:.4f} ms")
    # print(f"Cobra JIT:        {cobra_time:.4f} ms")

if __name__ == "__main__":
    main()