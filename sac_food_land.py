from braindance.games.food_land import EgocentricFoodSpikeEnv
import matplotlib.pyplot as plt
# conda create -n rl python=3.11
# conda activate rl
# pip install stable-baselines3
# pip install pygame


from stable_baselines3 import SAC
import math
import random
import time
from typing import Optional, Tuple, Union

import numpy as np
import pygame
from pygame import gfxdraw
import gym
from gym import spaces
from collections import deque



# We create the environment
# env = EgocentricFoodSpikeEnv(render_mode="human", use_history_observation=True, hunger=0)





env = EgocentricFoodSpikeEnv(render_mode="human", use_history_observation=True, hunger=0, reward_type='sparse', max_steps=500)
# env.step = step




model = SAC("MlpPolicy", env, verbose=1, learning_rate=.01,batch_size=256, buffer_size=100000)
print("Begin learning")
model.learn(total_timesteps=1000000, log_interval=4)

# rewards = []
# reward_trials = []
# # Step throuhg model manually
# obs = env.reset()
# for i in range(100000):
#     action, _states = model.predict(obs)#, deterministic=True)
#     obs, reward, dones, info = env.step(action)

#     rewards.append(reward)
#     # env.render()
#     if dones:

#         obs = env.reset()
#         reward_trials.append(sum(rewards))
#         rewards = []
#         print("~"*50)
#         print("Trial complete")
#         print("Reward: ", reward_trials[-1])

# plt.plot(reward_trials)
# plt.show()