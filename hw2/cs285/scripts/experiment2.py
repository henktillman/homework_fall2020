import os
import time

t0 = time.time()


for batch_size in [500, 1000, 5000, 10000]:
    for lr in [1e-3, 5e-3, 0.01, 0.02]:
        os.system(f"python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 3 -s 64 -b {batch_size} -lr {lr} -rtg --exp_name q2_b{batch_size}_r{lr}")

t1 = time.time()

print("Total experiment time elapsed: ", t1 - t0)
