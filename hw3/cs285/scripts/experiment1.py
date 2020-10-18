import os
import time

t0 = time.time()

os.system("python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1")
os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1")
os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2")
os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3")

os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 -- double_q --seed 1")
os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 -- double_q --seed 2")
os.system("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 -- double_q --seed 3")

t1 = time.time()

print("Total experiment time elapsed: ", t1 - t0)