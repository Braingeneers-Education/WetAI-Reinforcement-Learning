from cartpole_continuous import CartPoleContinuousEnv, CartPoleVectorEnv
import matplotlib.pyplot as plt
# conda create -n rl python=3.11
# conda activate rl
# pip install stable-baselines3
# pip install pygame

import numpy as np

from stable_baselines3 import SAC
# from stable_baselines3.common.logger import configure

# tmp_path = "/tmp/sb3_log/"
# set up logger
# new_logger = configure(tmp_path, ["stdout", "csv"])#], "tensorboard"])



# We create the environment
# env = EgocentricFoodSpikeEnv(render_mode="human", use_history_observation=True, hunger=0)

env = CartPoleVectorEnv(render_mode='human', num_envs=1)

model = SAC("MlpPolicy", env, verbose=1, learning_rate=.001,batch_size=256, buffer_size=100000)
# model.set_logger(new_logger)

model.learn(total_timesteps=1000000, log_interval=4)

rewards = []
reward_trials = []



# Step throuhg model manually



# obs,_ = env.reset()

# print(obs)
# for i in range(100000):
#     action, _states = model.predict(np.array(obs), deterministic=False)
#     obs, reward, dones, term,info = env.step(action[0])
#     model.train(16)
#     # print("Obs before", obs)
#     # obs = obs[:,0]

#     rewards.append(reward)
#     # env.render()
#     if dones:

#         obs_ = env.reset()
#         reward_trials.append(sum(rewards))
#         rewards = []
#         print("~"*50)
#         print("Trial complete")
#         print("Reward: ", reward_trials[-1])

# plt.plot(reward_trials)
# plt.show()