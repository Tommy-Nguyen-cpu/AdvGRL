import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import random
import requests

logPath = './rewards.log'

class AdvGRLEnv(gym.Env):
    def __init__(self, OriginalImagePath, pretrained_model, target_class, noise_amount, output_img_path="output.png", output_img_noise_path="output_noise.png", image_shape=(1080,1080,3)):
        super(AdvGRLEnv, self).__init__()

        self.labels = self.get_labels()
        self.original_img_path = OriginalImagePath
        self.adv_output = output_img_path
        self.adv_noise_output = output_img_noise_path
        self.feature_grid = np.array(Image.open(OriginalImagePath).resize((1080,1080)))
        self.min_cof = .01


        self.model = pretrained_model  # Load CLIP model
        self.target = target_class
        self.noiseAmount = noise_amount
        self.noise_space = None
        self.epochs = 0

        self.min_perturb = 0
        self.max_perturb = 255

        # Define action and observation spaces
        self.action_space = spaces.Box(low=self.min_perturb, high=self.max_perturb, shape=(image_shape), dtype=np.float16)  # Define space of allowable feature grid modifications
        self.observation_space = spaces.Box(low=0, high=255, shape=(image_shape), dtype=np.uint8) # Define state representation (e.g., feature grid, image)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset feature grid or load new one
        # self.noise_space = self.np_random.normal(scale=self.noiseAmount, size=self.action_space.shape)
        new_image = Image.open(self.original_img_path).resize((1080,1080))
        self.feature_grid = (np.array(new_image)).astype(np.uint8)
        new_image.close()

        return (self.feature_grid, {})
    
    def render(self):
        pass

    def close(self):
        pass

    def get_labels(self):

        # Download the ImageNet labels file
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        labels = response.json()

        labels.append('christmas tree')
        return labels

    def set_my_functions(self, reward_function, step_function):
        self.get_reward = reward_function
        self.step = step_function
