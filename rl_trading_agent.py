# Reinforcement Learning Trading Agent
# Single-file implementation: Custom Gym env, DQN agent, data handling, analytics, Streamlit dashboard

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Optional Streamlit dashboard
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# =============================
# 1. ENVIRONMENT
# =============================
class TradingEnv:
    def __init__(self, prices, window_size=30, transaction_cost=0.001):
        self.prices = np.array(prices)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.action_space = 3  # 0: Hold, 1: Buy, 2: Sell
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0
        self.done = False
        self.equity = 1.0
        self.equity_curve = [self.equity]
        self.actions = []
        self.returns_history = []
        return self._get_state()

    def _get_state(self):
        # State: window of returns
        return self.returns[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        reward = 0
        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        done = False

        # Action: 0=Hold, 1=Buy, 2=Sell
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                reward -= self.transaction_cost
            elif self.position == -1:
                # Close short, open long
                reward += (self.entry_price - price) / self.entry_price - self.transaction_cost
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                reward -= self.transaction_cost
            elif self.position == 1:
                # Close long, open short
                reward += (price - self.entry_price) / self.entry_price - self.transaction_cost
                self.position = -1
                self.entry_price = price
        else:  # Hold
            if self.position == 1:
                reward += (price - prev_price) / prev_price
            elif self.position == -1:
                reward += (prev_price - price) / prev_price

        self.current_step += 1
        self.actions.append(action)
        self.returns_history.append(reward)
        self.equity *= (1 + reward)
        self.equity_curve.append(self.equity)

        if self.current_step >= len(self.prices) - 1:
            done = True
            self.done = True
            # Liquidate position
            if self.position != 0:
                final_price = self.prices[self.current_step]
                if self.position == 1:
                    reward += (final_price - self.entry_price) / self.entry_price - self.transaction_cost
                elif self.position == -1:
                    reward += (self.entry_price - final_price) / self.entry_price - self.transaction_cost
                self.position = 0
                self.returns_history[-1] += reward
                self.equity *= (1 + reward)
                self.equity_curve[-1] = self.equity

        return self._get_state(), reward, done, {}

# =============================
# 2. DATA HANDLING
# =============================
def load_csv_prices(file_path):
    df = pd.read_csv(file_path)
    # Try to find price column
    for col in ['Close', 'close', 'Price', 'Adj Close', 'price']:
        if col in df.columns:
            return df[col].values
    # Fallback: first numeric column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            return df[col].values
    raise ValueError('No price column found in CSV.')

def generate_synthetic_prices(length=1000, mu=0.0005, sigma=0.01, start=100):
    # Geometric Brownian Motion
    returns = np.random.normal(mu, sigma, length)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices)

# =============================
# 3. DQN AGENT
# =============================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.action_dim = action_dim
        self.update_steps = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def push(self, *args):
        self.replay_buffer.push(*args)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

# =============================
# 4. TRAINING LOOP
# =============================
def train_agent(env, agent, episodes=50, verbose=True, progress_callback=None):
    rewards_history = []
    equity_curves = []
    action_counts = np.zeros(env.action_space)
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            state = next_state
            total_reward += reward
            action_counts[action] += 1
        rewards_history.append(total_reward)
        equity_curves.append(env.equity_curve.copy())
        if progress_callback:
            progress_callback(ep+1, episodes, rewards_history, env.equity_curve, action_counts)
        if verbose:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.4f} | Epsilon: {agent.epsilon:.3f}")
    return rewards_history, equity_curves, action_counts

# =============================
# 5. ANALYTICS & PLOTS
# =============================
def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10,4))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Step')
    plt.ylabel('Equity')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_training_rewards(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.title('Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_action_distribution(action_counts):
    plt.figure(figsize=(6,4))
    plt.bar(['Hold','Buy','Sell'], action_counts)
    plt.title('Action Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def sharpe_ratio(returns, risk_free=0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free) / (returns.std() + 1e-8) * np.sqrt(252)

def max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()

# =============================
# 6. MAIN ENTRY (CLI or Streamlit)
# =============================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='RL Trading Agent')
    parser.add_argument('--csv', type=str, help='Path to CSV file with prices')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--transaction_cost', type=float, default=0.001)
    parser.add_argument('--streamlit', action='store_true', help='Launch Streamlit dashboard')
    args = parser.parse_args()

    if args.csv:
        prices = load_csv_prices(args.csv)
    else:
        prices = generate_synthetic_prices(length=1500)

    env = TradingEnv(prices, window_size=args.window, transaction_cost=args.transaction_cost)
    agent = DQNAgent(state_dim=args.window, action_dim=3, lr=args.lr)

    rewards, equity_curves, action_counts = train_agent(env, agent, episodes=args.episodes)

    print(f"Sharpe Ratio: {sharpe_ratio(env.returns_history):.4f}")
    print(f"Max Drawdown: {max_drawdown(env.equity_curve):.4f}")

    plot_equity_curve(env.equity_curve)
    plot_training_rewards(rewards)
    plot_action_distribution(action_counts)

if __name__ == '__main__':
    main()
    input("Press Enter to exit and close all graphs...")
