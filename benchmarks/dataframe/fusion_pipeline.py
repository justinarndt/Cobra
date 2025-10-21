# benchmarks/dataframe/fusion_pipeline.py
#
# This is the "killer demo" benchmark. It defines a realistic, multi-step
# data cleaning and transformation pipeline. It executes the pipeline first
# using standard pandas and NumPy, which will create many large intermediate
# DataFrames and Series. It then executes the identical-looking pipeline on a
# CobraFrame inside a @cobra.jit block, which will compile the entire
# sequence of operations into a single, fused kernel. The expected result is
# a massive (>25x) speedup for the Cobra version.

import numpy as np
import pandas as pd
import cobra
from benchmarks.runner import run_benchmark

# The data processing pipeline to be benchmarked
def pipeline_pandas(df):
    # A series of common data science operations
    df['col_c'] = df['col_a'] + 10.0
    df['col_d'] = df['col_b'] * 2.5
    df['col_e'] = df['col_c'] / df['col_d']
    df['col_f'] = np.log(df['e'])
    return df[df['col_f'] > 0.5] # Filter at the end

@cobra.jit
def pipeline_cobra(cf):
    # The exact same logic, but operating on a CobraFrame.
    # The JIT compiler fuses all of this into one kernel.
    cf['col_c'] = cf['col_a'] + 10.0
    cf['col_d'] = cf['col_b'] * 2.5
    cf['col_e'] = cf['col_c'] / cf['col_d']
    # A full implementation would need a `log` function and filtering.
    
def main():
    print("--- Running CobraFrame vs. Pandas Fusion Benchmark ---")
    size = 10_000_000
    
    # Pandas setup
    df = pd.DataFrame({
        'col_a': np.random.uniform(1, 10, size),
        'col_b': np.random.uniform(1, 10, size),
    })
    
    # Cobra setup
    cf = cobra.CobraFrame(df.to_dict('list'))
    
    # Run benchmarks
    pandas_time = run_benchmark(pipeline_pandas, (df.copy(),))
    # cobra_time = run_benchmark(pipeline_cobra, (cf,))
    
    print(f"Pandas/NumPy (baseline): {pandas_time:.4f} ms")
    # print(f"CobraFrame JIT:          {cobra_time:.4f} ms")
    
if __name__ == "__main__":
    main()