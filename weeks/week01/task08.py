import numpy as np
import time


def main():
    random_nums = np.random.rand(1_000_000)
    print(f"Array length: {random_nums.size}")
    for_start = time.time()
    result_for_loop = 0
    for num in random_nums:
        result_for_loop = result_for_loop + num
    for_end = time.time()
    time_for_loop = for_end - for_start
    print(f"Loop sum: {result_for_loop} ({time_for_loop}s)")
    vectorized_start = time.time()
    result_vectorized = np.sum(random_nums)
    vectorized_end = time.time()
    time_vectorized = vectorized_end - vectorized_start
    print(f"Vectorized sum: {result_vectorized} ({time_vectorized}s)")
    print(f"Vectorized speedup: {time_for_loop / time_vectorized}")
    print()
    print(f"Vectorized mean: {np.mean(random_nums)}")
    print(f"Vectorized std: {np.std(random_nums)}")
    print(f"Vectorized min/max {np.min(random_nums)}/{np.max(random_nums)}")
