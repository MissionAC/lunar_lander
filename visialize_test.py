# %% [markdown]
# # SB3 PPO Model Visualization and Testing
# This notebook allows you to load the trained models from the hyperparameter 
# tuning experiments and visualize their performance in the environment.

# %%
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

# Suppress ALSA audio errors/warnings typically found in Linux/WSL environments
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Import configuration to ensure consistency with training
import config 

# %%
# 1. Define the directory where models are saved and list them
models_dir = "results/models"

if os.path.exists(models_dir):
    available_models = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    
    if available_models:
        print("Available trained models:")
        for i, model_name in enumerate(available_models):
            print(f"[{i}] {model_name}")
    else:
        print(f"No '.zip' models found in {models_dir}.")
else:
    print(f"Directory '{models_dir}' not found. Please run the training script first.")
    available_models = []

# %%
# 2. Select and load a specific model
# Change this index to select a different model from the list above
# MODEL_INDEX = -1 # Default to the last one in the list

# if available_models:
#     selected_model_name = available_models[MODEL_INDEX]
#     selected_model_path = os.path.join(models_dir, selected_model_name)
    
#     print(f"Loading model: {selected_model_name}")
    
#     # Load the trained model, forcing CPU usage to avoid the MlpPolicy GPU warning
#     model = PPO.load(selected_model_path, device="cpu")
#     print("Model loaded successfully!")
# else:
#     print("Cannot proceed: No models to load.")


model = PPO.load("results/models/model_lr0.0003_g0.99_gae0.95_clip0.2_arch128_128_ep480_rew168.06.zip", device="cpu")
# %%
# 3. Visualize the trained agent in the environment
# We use render_mode="human" to open a local window and watch the agent play

# if available_models:
print(f"Initializing environment: {config.ENV_NAME} (Continuous: {config.IS_CONTINUOUS})")

# Create the environment with human rendering enabled
env = gym.make(
    config.ENV_NAME, 
    continuous=config.IS_CONTINUOUS, 
    render_mode="human"
)

# Number of test episodes to visualize
test_episodes = 3

for ep in range(test_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0.0
    
    while not (done or truncated):
        # Predict the action using the trained model
        # Setting deterministic=True typically yields better and more stable test performance
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment forward
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # Optional: Add a slight delay if the rendering is too fast
        time.sleep(0.01)
        
    print(f"Test Episode {ep + 1} | Total Reward: {episode_reward:.2f}")

# Close the environment rendering window
env.close()
print("Visualization completed.")