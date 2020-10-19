import os
import time

t0 = time.time()

lr_list = [0.001, 0.005, 0.1]
for i in range(len(lr_list)):
    lr = lr_list[i]
    os.system(f"python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --lander_final_lr {lr} --exp_name q3_hparam{i+1}")


t1 = time.time()

print("Total experiment time elapsed: ", t1 - t0)