import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Create environment
env = gym.make("Hopper-v5", render_mode="rgb_array")

# Create a callback to save the model every 5000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,                # save every 5000 environment steps
    save_path="./checkpoints/",    # directory where checkpoints are saved
    name_prefix="sac_hopper"       # filename prefix
)

# Create model with your chosen hyperparameters
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_starts=10000,
    use_sde=False,
    device='cuda:1',
)

# Train with the callback
model.learn(total_timesteps=1_000_000, log_interval=4, callback=checkpoint_callback)

# Save final model
model.save("sac_hopper_final")

# Reload the last saved model
del model
model = SAC.load("sac_hopper_final")

# Evaluate (collect rgb frames and render to video)
import imageio

video_frames = []
obs, info = env.reset()
while True:
    # Get rendered rgb array after reset or step
    frame = env.render()
    if frame is not None:
        video_frames.append(frame)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        # Save video after first episode
        break

# Write frames to video file
imageio.mimsave("hopper_eval.mp4", video_frames, fps=30)