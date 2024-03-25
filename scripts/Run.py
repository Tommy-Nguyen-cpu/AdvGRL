import AdvGRL
import ClipClassifier
from PIL import Image
import numpy as np
import stable_baselines3

clf = ClipClassifier.CLIP_Classifier()

# Create environment instance
env = AdvGRL.AdvGRLEnv("./AIArchPhotos/IMG_5616.jpg", clf, "sliding door", 1)


def reward(predictions):
  reward = 0
  targetConf = 0

  # Penalize agent for not predicting target class in top 1.
  if predictions[0][1] != env.target:
      reward -= .6
  else:
     reward += 100
  
  # Encourage agent to produce noise for specific class.
  for prediction in predictions:
    if prediction[1] == env.target:
      reward += prediction[2] * 10 # High reward for christmas tree with high confidence
      targetConf = prediction[2]
      break
    # print("predicted: " + str(prediction))

  reward -= .001 # Penalize agent for each step.
  return reward, targetConf


def step(action):

    print("Action mean: " + str(action.std()))
    action = action * 125.5

    noise = (action).astype(np.uint8)

    # env.noise_space += noise

    # Modify feature grid based on action
    # env.feature_grid +=  noise  # Apply action to feature grid
    # print("feature grid: " + str(env.feature_grid.mean()))

    # Saves image with noise.
    noisy_image = Image.fromarray((env.feature_grid + noise))
    noisy_image.save(env.adv_output)
    noisy_image.close()

    # Save image of just the noise.
    new_noise = Image.fromarray((noise).astype(np.uint8))
    new_noise.save(env.adv_noise_output)
    new_noise.close()



    # Classify image using CLIP
    decoded_prediction = env.model.predict(env.adv_output, env.labels, top=1001)

    print("predicted: " + str(decoded_prediction[0]))

    # Calculate reward based on adversarial success and image quality
    reward, targetConf = env.get_reward(decoded_prediction)  # Implement reward function

    print("reward: " + str(reward))

    # Determine if episode is done
    done = False

    if decoded_prediction[0][1] == env.target and reward > 0:
       done = True

       print("RESETTING ENVIRONMENT")

    truncated, info = False, {}

    info = {
        "Target" : env.target,
        "Predicted" : decoded_prediction[0][1],
        "Confidence" : decoded_prediction[0][2]
    }

    env.epochs += 1

    return env.feature_grid + noise, reward, done, truncated, info


env.set_my_functions(reward, step)


def train_agent():
    # TODO: Memory issue is still a problem. Have no idea why though.
    # Create RL agent
    n_actions = env.action_space.shape[-1]
    action_noise = stable_baselines3.common.noise.NormalActionNoise(mean=np.zeros(n_actions), sigma=env.noiseAmount * np.ones(n_actions))
    model = stable_baselines3.TD3('CnnPolicy', env, device="cpu", buffer_size=15000, action_noise=action_noise)
    # print("Model activated...")

    # checkpoint_callback = CheckpointCallback(save_freq=100, save_path=".",
    #                                  name_prefix='adv_results')
    
    # Image to see if noise produced works.
    image = np.array(Image.open(env.original_img_path).resize((1080,1080)))
    while True:
       
       # Train model.
       model = model.learn(total_timesteps=200)

       # Test model action
       prediction = model.predict(image)

       # Apply noise to image, save image, and close image.
       test_img = Image.fromarray((image + prediction[0]).astype(np.uint8))
       test_img.save("test_output.png")
       test_img.close()

       # Predict top 1.
       decoded_prediction = env.model.predict("test_output.png", env.labels, top=1)
       print("Model predicted: " + str(decoded_prediction[0]))

       # If the top 1 is the target class and has a high confidence, we are done!
       if decoded_prediction[0][1] == env.target and decoded_prediction[0][2] > .5:
          print("Done!")
          break

       print("Done with one learn session.")
    
    # TODO: Try running it with multiple agents, figure out how our step and reward function will access necessary parameters.

train_agent()
