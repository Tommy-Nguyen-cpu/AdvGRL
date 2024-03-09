import AdvGRL
import ClipClassifier
from PIL import Image
import numpy as np
import requests
from stable_baselines3 import PPO, A2C
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

clf = ClipClassifier.CLIP_Classifier()

# Create environment instance
env = AdvGRL.AdvGRLEnv("./AIArchPhotos/IMG_5616.jpg", clf, "payphone", .05)


def get_labels():

    # Download the ImageNet labels file
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    labels = response.json()

    labels.append('christmas tree')
    return labels

def reward(predictions):
  reward = 0
  targetConf = 0

  # Penalize agent for not predicting target class in top 1.
  if predictions[0][1] != env.target:
      reward -= 1
  
  # Encourage agent to produce noise for specific class.
  for prediction in predictions:
    if prediction[1] == env.target:
      reward += prediction[2] * 10 # High reward for christmas tree with high confidence
      targetConf = prediction[2]
      break
    # print("predicted: " + str(prediction))

  reward -= .0001 # Penalize agent for each step.
  return reward, targetConf


def step(action):
    # print("action mean: " + str(action.mean()))
    # print("action std: " + str(action.std()))

    action = action * 125.5

    noise = (action *  env.np_random.uniform(low=0, high=env.noiseAmount, size=env.action_space.shape)).astype(np.uint8)

    # Modify feature grid based on action
    env.feature_grid +=  noise  # Apply action to feature grid
    # print("feature grid: " + str(env.feature_grid.mean()))

    # Saves image with noise.
    Image.fromarray(env.feature_grid).save(env.adv_output)

    # Save image of just the noise.
    Image.fromarray(noise).save(env.adv_noise_output)



    # Classify image using CLIP
    decoded_prediction = env.model.predict(env.adv_output, get_labels(), top=1001)

    print("predicted: " + str(decoded_prediction[0]))

    # Calculate reward based on adversarial success and image quality
    reward, targetConf = env.get_reward(decoded_prediction)  # Implement reward function

    print("reward: " + str(reward))

    # Determine if episode is done
    done = True if targetConf > env.min_cof and reward > 0 else False

    if done == True:
       if env.min_cof < .7:
            env.min_cof += .05
       print("RESETTING ENVIRONMENT")

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
        "Predicted" : decoded_prediction[0][1],
        "Confidence" : decoded_prediction[0][2]
    }

    return env.feature_grid, reward, done, truncated, info


env.set_my_functions(reward, step)


def train_agent():
    temperature = 1.0
    # Create RL agent
    model = PPO('CnnPolicy', env, gamma=0.999)
    # print("Model activated...")

    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=".",
                                     name_prefix='adv_results')
    for i in range(100):
       model.learn(total_timesteps=5)
       print("Done with one learn session.")
    
    # TODO: Try running it with multiple agents, figure out how our step and reward function will access necessary parameters.

    # episode_reward = 0
    # obs, info = env.reset()
    # _states = None
    # max_epochs = 0

    # while True:
    #     # Controls exploration vs exploitation.
    #     if np.random.random() < temperature:
    #         # print("Exploring!")
    #         action = model.action_space.sample()
    #     else:
    #         action, _states = model.predict(obs, _states)

    #     temperature -= .001

    #     obs, reward, done, truncated, info = env.step(action)
    #     # model = model.learn(total_timesteps=1, log_interval=4)
    #     episode_reward += reward
    #     print("Current episode reward: " + str(episode_reward))

    #     if info['Target'] == info['Predicted'] and info['Confidence'] > .6:
    #         print("Episode reward:", episode_reward)
    #         break

    #     # Resets the environment if it has been 100 iterations and we still haven't reached the target class.
    #     if (max_epochs > 10000 and info['Predicted'] != info['Target'] ) or (info['Predicted'] == info['Target'] and info['Confidence'] > .5):
    #         print("resetting environment...")
    #         obs, info = env.reset()
    #         max_epochs = 0
    #         episode_reward = 0
    #         _states = None
    #         temperature = 1.0

    #     max_epochs += 1

train_agent()
