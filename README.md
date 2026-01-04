# Reinforcement Learning – CartPole (DQN & PPO)

This project implements and compares **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** agents to solve the CartPole-v1 environment using Gymnasium and PyTorch.

## 🚀 Algorithms Used
- **DQN** (implemented from scratch)
- **PPO** (Stable-Baselines3)

## 🧠 Key Concepts
- Reinforcement Learning
- Value-based vs Policy-gradient methods
- Experience Replay
- Target Networks
- PPO Clipped Objective
- Robustness under noise

## 📊 Results
| Algorithm | Max Reward |
|---------|-----------|
| DQN | ~500 |
| PPO | 500 (stable) |

PPO converges faster and shows better stability compared to DQN.

## 🎥 Demo
A demo GIF is generated automatically after training:

videos/ppo_cartpole_demo_YYYYMMDD.gif
videos/dqn_cartpole_demo_YYYYMMDD.gif


## 🛠️ Installation
```bash
pip install -r requirements.txt

1. python train_dqn.py
2. python train_ppo.py
