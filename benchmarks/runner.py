# benchmarks/runner.py
#
# A generic utility for running and timing benchmark functions.
# This runner provides a consistent methodology for all our benchmarks,
# handling warm-up iterations to exclude compilation/caching overhead
# from measurements and calculating median execution time to provide
# stable performance numbers resistant to system noise.

import time
import numpy as np

def run_benchmark(func, args, num_warmup=5, num_iter=20):
    """
    Runs a given function with arguments and measures its performance.

    Args:
        func: The function to benchmark.
        args: A tuple of arguments to pass to the function.
        num_warmup (int): Number of warm-up runs before timing.
        num_iter (int): Number of timed iterations.

    Returns:
        The median execution time in milliseconds.
    """
    # Warm-up runs
    for _ in range(num_warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(num_iter):
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000) # Store in ms

    return np.median(times)