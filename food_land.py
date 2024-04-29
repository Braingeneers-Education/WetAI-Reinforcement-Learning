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


class EgocentricFoodSpikeEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 10000,
    }

    def __init__(self, render_mode: Optional[str] = None, reward_type = 'dense', hunger = .5, history=3,
                 use_history_observation= False, max_steps=100):
        """
        render_mode: str
            The mode to render the environment in. Can be "human" or None.
        reward_type: str
            The type of reward to use. Can be "sparse" or "dense".
            sparse only gives rewards for getting food, while dense gives rewards for getting 
                food, getting close to food, and penalizes for getting close to spikes.
        hunger: float
            The rate at which the agent gets hungry. 
            The higher the value, the faster the agent gets hungry. Hunger is a linear bias making agents not want
            to sit around and do nothing when close to food. 
        history: int
            The number of steps to keep in the observation history. 
            This is used for the RL agent to have a sense of the environment's dynamics.

        use_history_observation: bool
            Whether to include the history of observations in the observation space. 
            This is needed for RL agents, but not for organoids.
        max_steps: int
            The maximum number of steps before the environment resets.


        """
        self.screen_width = 800
        self.screen_height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        self.max_steps = max_steps
        self.step_count = 0

        self.reward_type = reward_type

        self.agent_pos = np.array([400, 300], dtype=np.float32)
        self.agent_dir = 0  # Agent's direction in radians


        self.food_positions = []
        self.food_count = 3
        self.spike_count = 0
        self.food_signal = 0
        self.spike_signal = 0
        self.signal_range = 100
        self.spike_positions = []

        self.hunger = hunger
        self.food_got = 0
        self.spike_hit = 0

        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2)

        self.use_history_observation = use_history_observation

        self.action_shape = (2,)
        self.action_space = spaces.Box(
            low=np.array([-1,0]), high=np.array([1,1]), shape=(2,), dtype=np.float32
        )
        if self.use_history_observation:
            self.observation_shape = (4*history + 2*(history - 1),)
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=self.observation_shape, dtype=np.float32
            )
        else:
            self.observation_shape = (2,)
            self.observation_space = spaces.Box(
                low=-1, high=2, shape=(2,), dtype=np.float32
            )

        self.observation_history = deque(maxlen=history)
        self.action_history = deque(maxlen=history)
        # Fill both with zeros
        for _ in range(3):
            self.observation_history.append(np.zeros(2, dtype=np.float32))
            self.action_history.append(np.zeros(2, dtype=np.float32))


        # Rendering
        self.draw_food_ani = 0
        self.draw_spike_ani = 0

        self.generate_food()
        self.generate_spikes()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # if action[0]:  # Turn left
        #     self.agent_dir += 0.2
        # if action[1]:  # Turn right
        #     self.agent_dir -= 0.2
        # if action[2]:  # Move forward
        #     self.agent_pos[0] += 8 * math.cos(self.agent_dir)
        #     self.agent_pos[1] += 8 * math.sin(self.agent_dir)
        # if action[3]:  # Move backward
        #     self.agent_pos[0] -= 8 * math.cos(self.agent_dir)
        #     self.agent_pos[1] -= 8 * math.sin(self.agent_dir)
        self.agent_dir += action[0]*.2

        self.agent_pos[0] += 8 * math.cos(self.agent_dir)*action[1]
        self.agent_pos[1] += 8 * math.sin(self.agent_dir)*action[1]


        self.agent_pos = np.clip(self.agent_pos, 0, [self.screen_width, self.screen_height])

        self.food_signal = 0
        self.spike_signal = 0
        self.food_got = 0
        self.spike_hit = 0

        for i, food_pos in enumerate(self.food_positions):
            dist = np.linalg.norm(self.agent_pos - food_pos)
            if dist < 20:
                del self.food_positions[i]
                self.generate_food()
                self.food_got += 1

            else:
                # self.food_signal += 1 / (dist**2 / self.max_distance**2)
                self.food_signal += 1 / (dist**2 / self.signal_range**2)

        for i, spike_pos in enumerate(self.spike_positions):
            dist = np.linalg.norm(self.agent_pos - spike_pos)
            self.spike_signal += 1 / (dist**2 / self.signal_range**2)
            if dist < 20:
                # Ouch! We hit a spike
                self.spike_hit += 1
                del self.spike_positions[i]
                self.generate_spikes()
            

        # Normalize them 
        self.food_signal /= 10
        self.spike_signal /= 10

        observation = np.array([self.food_signal, self.spike_signal], dtype=np.float32)
        # reward = self.food_signal - self.spike_signal
        reward = self.calculate_reward(self.food_got, self.spike_hit)
        done = True if self.step_count >= self.max_steps else False
        info = {}

        if self.render_mode == "human":
            self.render()

        self.observation_history.append(observation)
        self.action_history.append(action)

        # Return the last 3 observations and actions as the observation
        observation = self.get_observation(observation)
        self.step_count += 1
        return observation, reward, done, info
    
    def get_observation(self, observation):
        """ Concatenates history, needed for RL, but not for organoid"""
        if self.observation_history:
            observation_all = np.concatenate(self.observation_history)
            obs_diff = np.diff(np.array(self.observation_history), axis=0)
            # Flatten the difference
            obs_diff = obs_diff.flatten()
            # 
            action_all = np.concatenate(self.action_history)
            observation = np.concatenate([observation_all, obs_diff, action_all])

        return observation
    

    def calculate_reward(self, food_got, spike_hit):
        if self.reward_type == 'sparse':
            return food_got - spike_hit*5
        elif self.reward_type == 'dense':
            reward = (self.food_signal - self.spike_signal) + food_got*10 - spike_hit*20 - self.hunger
            
            return reward

    def reset(self):
        self.agent_pos = np.array([400, 300], dtype=np.float32)
        self.agent_dir = 0
        self.food_positions = []
        self.spike_positions = []
        self.generate_food()
        self.generate_spikes()
        self.step_count = 0
        self.food_signal = 0
        self.spike_signal = 0
        self.food_got = 0
        self.spike_hit = 0
        
        # Concat observation history and action history
        info = {}

        if self.render_mode == "human":
            self.render()

        observation = np.array([self.food_signal, self.spike_signal], dtype=np.float32)
        observation = self.get_observation(observation)

        return observation#, info

    def generate_food(self):
        while len(self.food_positions) < self.food_count:
            food_pos = np.random.randint(0, [self.screen_width, self.screen_height])
            if not any(np.all(food_pos == pos) for pos in self.food_positions):
                self.food_positions.append(food_pos)

    def generate_spikes(self):
        while len(self.spike_positions) < self.spike_count:
            spike_pos = np.random.randint(0, [self.screen_width, self.screen_height])
            if not any(np.all(spike_pos == pos) for pos in self.spike_positions):
                self.spike_positions.append(spike_pos)

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.font.init()
                self.my_font = pygame.font.SysFont('Comic Sans MS', 30)
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.surf.fill((255, 255, 255))

        for food_pos in self.food_positions:
            pygame.draw.circle(self.surf, (0, 255, 0), food_pos, 10)
            
            # Draw signal range circle
            # Adjust this value to change the signal range
            signal_color = (0, 255, 0, 60)  # Green color with transparency (RGBA format)
            pygame.draw.circle(self.surf, signal_color, food_pos, self.signal_range, 2)  # Draw the circle with a thickness of 2

        for spike_pos in self.spike_positions:
            signal_color = (100, 100 , 100, 40) 
            pygame.draw.polygon(self.surf, (128, 128, 128), [
                (spike_pos[0], spike_pos[1] - 15),
                (spike_pos[0] - 10, spike_pos[1] + 10),
                (spike_pos[0] + 10, spike_pos[1] + 10)
            ])
            pygame.draw.circle(self.surf, signal_color, spike_pos, self.signal_range, 2)  # Draw the circle with a thickness of 2


        # Draw a more detailed mouse shape for the head
        mouse_color = (150, 75, 0)  # Brown color for the mouse
        ear_color = (130, 50, 0)  # Red color for the ears
        mouse_size = 36
        mouse_surface = pygame.Surface((mouse_size, mouse_size), pygame.SRCALPHA)
        # pygame.draw.circle(mouse_surface, mouse_color, (mouse_size//2, mouse_size//2),mouse_size//2.4)
        # Ellispe
        pygame.draw.ellipse(mouse_surface, mouse_color, (5, 0, mouse_size-10, mouse_size))

        
        # Draw mouse ears
        ear_size = 6
        ear_offset = 14
        ear_left = (mouse_size // 2 - ear_offset, mouse_size//2)
        ear_right = (mouse_size // 2 + ear_offset, mouse_size//2)
        pygame.draw.circle(mouse_surface, ear_color, ear_left, ear_size)
        pygame.draw.circle(mouse_surface, ear_color, ear_right, ear_size)
        
        # Draw mouse eyes
        eye_size = 2
        eye_offset = 6
        eye_left = (mouse_size // 2 - eye_offset, mouse_size // 2 + mouse_size // 4 - eye_size // 2)
        eye_right = (mouse_size // 2 + eye_offset, mouse_size // 2 + mouse_size // 4 - eye_size // 2)
        pygame.draw.circle(mouse_surface, (0, 0, 0), eye_left, eye_size)
        pygame.draw.circle(mouse_surface, (0, 0, 0), eye_right, eye_size)
        
        # Draw mouse nose
        nose_size = 4
        nose_pos = (mouse_size // 2, mouse_size // 2 + mouse_size // 3)
        pygame.draw.circle(mouse_surface, (255, 192, 203), nose_pos, nose_size)

        # Draw mouse tail
        tail_length = 20
        tail_width = 1
        tail_color = (150, 75, 0)
        tail_start = (self.agent_pos - 20 * np.array([math.cos(self.agent_dir), math.sin(self.agent_dir)])).astype(int)
        tail_end = (tail_start - tail_length * np.array([math.cos(self.agent_dir), math.sin(self.agent_dir)])).astype(int)
        # Make it curve
        control_point = (tail_start + tail_end) / 2
        control_point[1] += 20
        # pygame.draw.line(self.surf, tail_color, tail_start, tail_end, tail_width)
        gfxdraw.bezier(self.surf, [tail_start, control_point, tail_end], 3, tail_color)




        
        # Rotate the mouse surface based on the agent's direction
        rotated_mouse_surface = pygame.transform.rotate(mouse_surface, -math.degrees(self.agent_dir) + 90)
        mouse_rect = rotated_mouse_surface.get_rect(center=tuple(self.agent_pos.astype(int)))
        self.surf.blit(rotated_mouse_surface, mouse_rect)

        pygame.draw.line(self.surf, (0, 0, 255), tuple(self.agent_pos.astype(int)),
                        tuple((self.agent_pos + 20 * np.array([math.cos(self.agent_dir), math.sin(self.agent_dir)])).astype(int)), 3)


        # If food_count >= 0, draw a animation of got food
        if self.food_got > 0:
            self.draw_food_ani = 20

        if self.draw_food_ani > 0:
            text_surface = self.my_font.render(f'Got food!', False, (40, 200, 40))
            # Halfway between the screen width and the text width
            self.surf.blit(text_surface, (self.screen_width//2 - text_surface.get_width()//2, 20))
            self.draw_food_ani -= 1

        # If spike_count > 0, draw a animation of hit spike
        if self.spike_hit > 0:
            self.draw_spike_ani = 20

        if self.draw_spike_ani > 0:
            text_surface = self.my_font.render(f'Hit spikes :(!', False, (200, 40, 40))
            # Halfway between the screen width and the text width
            self.surf.blit(text_surface, (self.screen_width//2 - text_surface.get_width()//2, 20))
            self.draw_spike_ani -= 1



        text_surface = self.my_font.render(f'{self.food_signal}', False, (40, 200, 40))
        # Spike text
        text_surface2 = self.my_font.render(f'{self.spike_signal}', False, (100, 100, 100))

        self.screen.blit(self.surf, (0, 0))
        self.screen.blit(text_surface, (0, 0))
        self.screen.blit(text_surface2, (0, 30))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    env = EgocentricFoodSpikeEnv(render_mode="human")
    env.reset()

    cur_reward = 0
    rewards = []

    action = np.zeros(2, dtype=np.float32)

    while True:
        time.sleep(0.05)
        action = np.zeros(2, dtype=np.float32)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action[0] += 1
        if keys[pygame.K_LEFT]:
            action[0] -= 1
        if keys[pygame.K_UP]:
            action[1] += 1
        if keys[pygame.K_DOWN]:
            action[1] -= 1

        observation, reward, done, info = env.step(action)
        cur_reward += reward

        env.render()

        if done:
            print(f"Episode finished. Reward: {cur_reward}")
            rewards.append(cur_reward)
            print("All rewards:", rewards)
            cur_reward = 0
            env.reset()