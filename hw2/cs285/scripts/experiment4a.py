import os
import time

t0 = time.time()


for batch_size in [10000, 30000, 50000]:
    for lr in [0.005, 0.01, 0.02]:
        os.system(f"python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 3 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline \
--exp_name q4_search_b{batch_size}_lr{lr}_rtg_nnbaseline")

t1 = time.time()

print("Total experiment time elapsed: ", t1 - t0)


# python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 0.1 -rtg \
# --exp_name q2_b500_r0.1