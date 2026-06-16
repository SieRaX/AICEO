import os
# classic control(pygame) 렌더링을 화면 없이 수행하기 위한 설정.
# gymnasium 을 import 하기 전에 설정해야 합니다.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Create environment (원본 MountainCar-v0)
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# Create a callback to save the model every 20000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=20000,                  # save every 20000 environment steps
    save_path="./checkpoints/",       # directory where checkpoints are saved
    name_prefix="dqn_mountaincar",    # filename prefix
)

# Create model with hyperparameters tuned for MountainCar-v0
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=4e-3,
    batch_size=128,
    buffer_size=10000,
    learning_starts=1000,
    gamma=0.98,
    target_update_interval=600,
    train_freq=16,
    gradient_steps=8,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    device="auto",                    # MountainCar는 작아서 CPU로도 충분합니다.
)

# Train with the callback
model.learn(total_timesteps=200_000, log_interval=10, callback=checkpoint_callback)

# Save final model
model.save("dqn_mountaincar_final")

# Reload the last saved model
del model
model = DQN.load("dqn_mountaincar_final")

# Evaluate (collect rgb frames and render to video)
import imageio
video_frames = []
obs, info = env.reset(seed=42)
cumreward = 0
while True:
    # Get rendered rgb array after reset or step
    frame = env.render()
    if frame is not None:
        video_frames.append(frame)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    cumreward += reward
    if terminated or truncated:
        # Save video after first episode
        break

# Write frames to video file
imageio.mimsave("mountaincar_eval.mp4", video_frames, fps=30)
print(f"[EVAL] total reward = {cumreward}")
print("DONE")