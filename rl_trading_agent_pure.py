import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 1. Custom Trading Environment
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
        self.equity = 1.0
        self.equity_curve = [self.equity]
        self.actions = []
        self.returns_history = []
        return self._get_state()

    def _get_state(self):
        # Return window_size-1 returns for DQN input
        return self.returns[self.current_step - self.window_size + 1:self.current_step]

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
                reward += (self.entry_price - price) / self.entry_price - self.transaction_cost
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                reward -= self.transaction_cost
            elif self.position == 1:
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

# 2. DQN Agent
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

# 3. Synthetic Price Data (GBM)
def generate_synthetic_prices(length=1000, mu=0.0005, sigma=0.01, start=100):
    returns = np.random.normal(mu, sigma, length)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices)

def main():
    window_size = 30
    episodes = 50
    transaction_cost = 0.001
    lr = 1e-3
    prices = generate_synthetic_prices(length=1000)
    env = TradingEnv(prices, window_size=window_size, transaction_cost=transaction_cost)
    agent = DQNAgent(state_dim=window_size-1, action_dim=3, lr=lr)
    reward_history = []
    equity_history = []
    action_counts = np.zeros(3)
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            action_counts[action] += 1
        reward_history.append(total_reward)
        equity_history.append(env.equity_curve[-1])
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.4f} | Epsilon: {agent.epsilon:.3f}")
    # 1) Price chart
    plt.figure()
    plt.plot(prices)
    plt.title("Synthetic Price Chart")
    plt.xlabel("Step")
    plt.ylabel("Price")
    # 2) Equity curve
    plt.figure()
    plt.plot(env.equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    # 3) Training reward curve
    plt.figure()
    plt.plot(reward_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    # 4) Action distribution
    plt.figure()
    plt.bar(["Hold", "Buy", "Sell"], action_counts)
    plt.title("Action Distribution")
    plt.ylabel("Count")
    # Show all figures, block until closed
    plt.show(block=True)
    input("Press Enter to exit and close all graphs...")

if __name__ == "__main__":
    main()
