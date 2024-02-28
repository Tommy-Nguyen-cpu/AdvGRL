import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import random

logPath = './rewards.log'

class AdvGRLEnv(gym.Env):
    def __init__(self, OriginalImagePath, pretrained_model, target_class, noise_amount, output_img_path="output.jpg", output_img_noise_path="output_noise.jpg", image_shape=(1080,1080,3)):
        super(AdvGRLEnv, self).__init__()

        self.original_img_path = OriginalImagePath
        self.adv_output = output_img_path
        self.adv_noise_output = output_img_noise_path
        self.feature_grid = np.array(Image.open(OriginalImagePath))
        self.log = open(logPath, 'a')


        self.model = pretrained_model  # Load CLIP model
        self.target = target_class
        self.noiseAmount = noise_amount
        self.max_reward = -np.inf

        self.state = 38 + random.randint(-3,3)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0, high=255, shape=(image_shape), dtype=np.uint8)  # Define space of allowable feature grid modifications
        self.observation_space = spaces.Box(low=0, high=255, shape=(image_shape), dtype=np.uint8) # Define state representation (e.g., feature grid, image)

    def reset(self, seed=0):
        super().reset(seed=seed)
        # Reset feature grid or load new one
        self.feature_grid = np.array(Image.open(self.original_img_path))
        # self.log.close()
        return (self.feature_grid, {})
    
    def render(self):
        pass

    def close(self):
        self.log.close()


    def set_functions(self, reward_function, step_function):
        self.get_reward = reward_function
        self.step = step_function
    
    def get_target(self):
        return self.target

    def get_observation(self):
        return self.feature_grid
    
    def get_og_path(self):
        return self.original_img_path
    
    def get_adv_img_path(self):
        return self.adv_output
    
    def get_adv_noise(self):
        return self.adv_noise_output
    
    def get_classifier(self):
        return self.model
    
    def get_action_space_shape(self):
        return self.action_space.shape
