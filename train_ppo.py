"""
train_ppo_full_unstable.py

Full PPO project for CartPole-v1
- Train PPO
- Evaluate PPO
- Record demo GIF
- FORCE instability (varying rewards)

Usage:
    python train_ppo_full_unstable.py

Outputs:
 - ppo_cartpole.zip
 - videos/ppo_cartpole_demo_*.gif
"""

# =========================
# Imports
# =========================
import os
import datetime
import numpy as np
import imageio
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# =========================
# Config
# =========================
ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 100_000
SOLVED_SCORE = 475

VIDEO_DIR = "videos"
NUM_DEMO_EPISODES = 3
MAX_STEPS = 500

# --- FORCE INSTABILITY ---
OBS_NOISE_STD = 0.10          # HIGH noise → guaranteed instability
RANDOM_ACTION_PROB = 0.10     # 10% forced random actions

os.makedirs(VIDEO_DIR, exist_ok=True)

# =========================
# Train PPO
# =========================
def train_ppo():
    print("Training PPO agent...")

    env = Monitor(gym.make(ENV_ID))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_cartpole")

    env.close()
    print("PPO training finished and model saved.")

    return model

# =========================
# Evaluate PPO (stable)
# =========================
def evaluate_ppo(model):
    print("Evaluating PPO agent...")

    eval_env = Monitor(gym.make(ENV_ID))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    eval_env.close()

    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward

# =========================
# Record UNSTABLE Demo GIF
# =========================
def record_demo_ppo_unstable(model):
    print("Recording UNSTABLE PPO demo...")

    env = gym.make(ENV_ID, render_mode="rgb_array")
    frames = []

    for ep in range(NUM_DEMO_EPISODES):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < MAX_STEPS:
            # Add observation noise
            obs = obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Stochastic action
            action, _ = model.predict(obs, deterministic=False)

            # Force random action sometimes
            if np.random.rand() < RANDOM_ACTION_PROB:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Recorded episode {ep}, reward = {total_reward}")

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(VIDEO_DIR, f"ppo_cartpole_demo_{timestamp}.gif")

    imageio.mimsave(gif_path, frames, fps=30)
    print(f"PPO demo GIF saved to {gif_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    model = train_ppo()

    mean_reward = evaluate_ppo(model)

    if mean_reward >= SOLVED_SCORE:
        print("Environment solved!")
        record_demo_ppo_unstable(model)
    else:
        print("Environment not solved yet. Demo skipped.")
