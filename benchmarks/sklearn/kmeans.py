# benchmarks/sklearn/kmeans.py
#
# Benchmarks the performance of KMeans clustering, comparing the stock
# scikit-learn implementation against the Cobra-patched version to
# demonstrate the performance gains from JIT compilation.

import time
import cobra
import numpy as np
from sklearn.datasets import make_blobs

# --- 1. Generate a large synthetic dataset ---
print("Generating synthetic data for KMeans benchmark...")
n_samples = 1_000_000
n_features = 32
n_clusters = 16
data, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=42
)
print(f"Data shape: {data.shape}")

# --- 2. Benchmark the stock Scikit-learn implementation ---
from sklearn.cluster import KMeans

print("\n--- Benchmarking stock Scikit-learn KMeans ---")
start_time_stock = time.perf_counter()

stock_kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=100)
stock_kmeans.fit(data)

end_time_stock = time.perf_counter()
duration_stock = end_time_stock - start_time_stock
print(f"Stock scikit-learn time: {duration_stock:.4f} seconds")


# --- 3. Apply the patch and benchmark the Cobra-accelerated version ---
print("\n--- Benchmarking Cobra-patched Scikit-learn KMeans ---")

# Apply the monkey-patch
cobra.patch_sklearn()

# Re-import KMeans. This time, Python will give us our PatchedKMeans class.
from sklearn.cluster import KMeans as PatchedKMeans

start_time_patched = time.perf_counter()

patched_kmeans = PatchedKMeans(n_clusters=n_clusters, n_init=1, max_iter=100)
patched_kmeans.fit(data)

end_time_patched = time.perf_counter()
duration_patched = end_time_patched - start_time_patched
print(f"Patched scikit-learn time: {duration_patched:.4f} seconds")

# --- 4. Report the results ---
speedup = duration_stock / duration_patched if duration_patched > 0 else float('inf')
print("\n" + "="*40)
print(f"Performance Speedup: {speedup:.2f}x")
print("="*40)