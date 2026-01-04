"""
train_dqn.py
A complete, runnable DQN implementation for CartPole-v1 using gymnasium + PyTorch.

Usage:
    python train_dqn.py

Outputs:
 - Trained model saved as 'dqn_cartpole.pth'
 - Training reward plot 'training_rewards.png'
 - Optional recorded videos inside ./videos/
"""

import os
import random
import math
from collections import deque, namedtuple
from dataclasses import dataclass
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import RecordVideo  # for video recording

# -------------------------
# Config / Hyperparameters
# -------------------------
@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100000
    min_replay_size: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frames: int = 50_000  # decay over frames
    target_update_every: int = 1000  # steps
    max_frames: int = 150_000  # total environment steps to run
    eval_every: int = 5000
    eval_episodes: int = 10
    hidden_dim: int = 128
    save_path: str = "dqn_cartpole.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    video_dir: str = "videos"
    record_video_when_solved: bool = True
    solved_score: float = 475.0  # CartPole-v1 max is 500
    print_every: int = 500

cfg = Config()
print(f"Using device: {cfg.device}")

# -------------------------
# Utilities
# -------------------------
def set_seed(env, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except Exception:
        pass

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors
        states = torch.tensor(np.vstack([b.state for b in batch]), dtype=torch.float32, device=cfg.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=cfg.device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=cfg.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack([b.next_state for b in batch]), dtype=torch.float32, device=cfg.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=cfg.device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Q-network
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.q_net = QNetwork(obs_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.target_q_net = QNetwork(obs_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.buffer_size)
        self.n_actions = n_actions
        self.total_steps = 0

    def select_action(self, state, epsilon):
        """
        state: numpy array (obs_dim,)
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # current Q
        q_values = self.q_net(states).gather(1, actions)

        # target Q
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1.0 - dones) * (self.cfg.gamma * next_q_values)

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cfg': vars(self.cfg)
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['q_state_dict'])
        self.target_q_net.load_state_dict(data['target_state_dict'])
        self.optimizer.load_state_dict(data['optimizer'])

# -------------------------
# Training loop
# -------------------------
def linear_epsilon(frame_idx):
    # linear decay from eps_start to eps_end over eps_decay_frames
    if frame_idx >= cfg.eps_decay_frames:
        return cfg.eps_end
    slope = (cfg.eps_end - cfg.eps_start) / cfg.eps_decay_frames
    return cfg.eps_start + slope * frame_idx

def evaluate_policy(env, agent, episodes=5, render=False):
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(obs, epsilon=0.0)  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            if render:
                env.render()
        returns.append(total)
    return np.mean(returns)

def train():
    os.makedirs(cfg.video_dir, exist_ok=True)
    env = gym.make(cfg.env_id)
    set_seed(env, cfg.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)

    # populate replay with random policy
    obs, info = env.reset()
    for _ in range(cfg.min_replay_size):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, info = env.reset()

    losses = []
    all_rewards = []
    episode_reward = 0
    obs, info = env.reset()
    episode = 0
    start_time = time.time()

    pbar = trange(cfg.max_frames, desc="Frames")
    for frame_idx in pbar:
        epsilon = linear_epsilon(frame_idx)
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += reward
        agent.total_steps += 1

        loss = agent.update(cfg.batch_size)
        if loss is not None:
            losses.append(loss)

        if done:
            all_rewards.append(episode_reward)
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_last10 = np.mean(all_rewards[-10:])
                pbar.set_postfix({
                    "episode": episode,
                    "avg_last10": f"{avg_last10:.2f}",
                    "eps": f"{epsilon:.3f}",
                    "loss": f"{np.mean(losses[-100:]):.4f}" if losses else "nan"
                })
            episode_reward = 0
            episode += 1
            obs, info = env.reset()

        # sync target
        if frame_idx % cfg.target_update_every == 0:
            agent.sync_target_network()

        # periodic evaluation
        if frame_idx % cfg.eval_every == 0 and frame_idx > 0:
            eval_env = gym.make(cfg.env_id)
            set_seed(eval_env, cfg.seed + 123)
            mean_return = evaluate_policy(eval_env, agent, episodes=cfg.eval_episodes)
            print(f"\nFrame {frame_idx} — Eval mean return: {mean_return:.2f}")
            eval_env.close()
            # optionally record a short video when solved
            if cfg.record_video_when_solved and mean_return >= cfg.solved_score:
                print("Solved! Recording a short video for demo...")
                record_demo_no_moviepy(agent)
                break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes, episodes: {episode}")

    # Save
    agent.save(cfg.save_path)
    print(f"Saved model to {cfg.save_path}")

    # Plot rewards
    plt.figure(figsize=(8,4))
    plt.plot(all_rewards, label="episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training rewards per episode")
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_rewards.png")
    print("Saved training plot to training_rewards.png")

    env.close()

# Add at top of file
import datetime
import imageio
import os

# Replace existing record_demo with this function:
def record_demo_no_moviepy(agent, num_episodes=3, max_steps=500):
    """
    Records episodes by creating an env with render_mode='rgb_array' and saving a GIF.
    Doesn't rely on gym's RecordVideo / moviepy.
    """
    # create a fresh env that returns RGB frames
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    os.makedirs(cfg.video_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(cfg.video_dir, f"dqn_cartpole_demo_{timestamp}.gif")
    all_frames = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        total = 0
        while not done and steps < max_steps:
            # get frame BEFORE or AFTER step? we'll render the current state
            frame = env.render()
            if frame is not None:
                all_frames.append(frame)
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        print(f"Recorded ep {ep} reward {total}")

    env.close()

    if len(all_frames) == 0:
        print("No frames captured; GIF not created.")
        return

    # Save GIF (imageio expects frames as uint8)
    imageio.mimsave(gif_path, all_frames, fps=30)
    print(f"Saved demo GIF to {gif_path}")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    train()
