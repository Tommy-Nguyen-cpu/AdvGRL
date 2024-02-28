import AdvGRL
import ClipClassifier
from PIL import Image
import numpy as np
import requests
from stable_baselines3 import PPO, A2C

clf = ClipClassifier.CLIP_Classifier()

# Create environment instance
env = AdvGRL.AdvGRLEnv("./instant-ngp/NoObjectScene/Screenshots/0009.jpg", clf, "christmas tree", .4)

def get_labels():

    # Download the ImageNet labels file
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    labels = response.json()

    labels.append('christmas tree')
    return labels

def reward(predictions):
    # TODO: Define reward function.
    reward = -10
    if predictions[1] == env.target:
        reward = predictions[2] * 10
    return reward

def step(action):

    noise = (action.squeeze() * np.random.normal(scale=env.noiseAmount, size=env.action_space.shape)).astype(np.uint8)
    # Modify feature grid based on action
    env.feature_grid +=  noise  # Apply action to feature grid

    # Saves image with noise.
    Image.fromarray(env.feature_grid).save(env.adv_output)

    # Save image of just the noise.
    Image.fromarray(noise).save(env.adv_noise_output)

    # Classify image using CLIP
    decoded_prediction = env.model.predict(env.adv_noise_output, get_labels(), top=1)[0]
    print("predicted: " + str(decoded_prediction))

    # Calculate reward based on adversarial success and image quality
    reward = env.get_reward(decoded_prediction)  # Implement reward function

    # Determine if episode is done
    done = False

    truncated, info = False, {}
    
    if decoded_prediction[1] == env.target:
    # if self.epochs % 250 == 0:
        print('saving tape...')
        env.log.write(
            '\n'
            + 
            + str(decoded_prediction)
        )

    info = {
        "Target" : env.target,
        "Predicted" : decoded_prediction[1],
        "Confidence" : decoded_prediction[2]
    }

    return env.feature_grid, reward, done, truncated, info


env.set_functions(reward, step)

# Create RL agent
model = PPO('MlpPolicy', env, device='cuda')

def train_agent():
    episode_reward = 0
    obs, info = env.reset()
    _states = None

    max_epochs = 0
    while True:
        action, _states = model.predict(obs, _states)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        print("Current episode reward: " + str(episode_reward))

        if info['Target'] == info['Predicted'] and info['Confidence'] > .6:
            print("Episode reward:", episode_reward)
            break

        # Resets the environment if it has been 100 iterations and we still haven't reached the target class.
        if max_epochs > 100 and info['Predicted'] != info['Target']:
            print("resetting environment...")
            obs, info = env.reset()
            max_epochs = 0
            episode_reward = 0
            _states = None
        
        # If we reached the target class, reset iteration counter.
        elif info['Predicted'] == info['Target']:
            max_epochs = 0

        max_epochs += 1

train_agent()
