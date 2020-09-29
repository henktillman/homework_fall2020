import os
import time

t0 = time.time()

batch = 30000
lr = 0.02


os.system(f"python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch} -lr {lr} \
--exp_name q4_b{batch}_r{lr}")
os.system(f"python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch} -lr {lr} -rtg \
--exp_name q4_b{batch}_r{lr}_rtg")
os.system(f"python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch} -lr {lr} --nn_baseline \
--exp_name q4_b{batch}_r{lr}_nnbaseline")
os.system(f"python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch} -lr {lr} -rtg --nn_baseline \
--exp_name q4_b{batch}_r{lr}_rtg_nnbaseline")

t1 = time.time()

print("Total experiment time elapsed: ", t1 - t0)
