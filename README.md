## Creator/Dev: tubakhxn

# Reinforcement Learning Trading Agent

This project implements a pure Python Deep Q-Learning trading agent for financial markets. It uses synthetic price data, a custom trading environment, and PyTorch for the agent. All graphs are displayed in pop-up windows using matplotlib.

## What is this project about?
- **Custom Trading Environment:** Simulates trading with rolling return windows, actions (Hold, Buy, Sell), transaction costs, and portfolio tracking.
- **Deep Q-Network (DQN):** Uses PyTorch for a simple feedforward neural network, experience replay, and epsilon-greedy exploration.
- **Synthetic Price Data:** Generates prices using Geometric Brownian Motion (GBM), so no external data/API is needed.
- **Pop-up Graphs:** Opens separate matplotlib windows for price chart, equity curve, training reward curve, and action distribution.

## How to fork and run
1. Fork this repository on GitHub.
2. Clone your fork:
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
3. Install dependencies:
   ```
   pip install numpy torch matplotlib
   ```
4. Run the script in your terminal:
   ```
   python rl_trading_agent_pure.py
   ```

## Related topics to explore
- [Reinforcement Learning (Wikipedia)](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Deep Q-Network (Wikipedia)](https://en.wikipedia.org/wiki/Deep_Q-network)
- [Geometric Brownian Motion (Wikipedia)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Financial Markets (Wikipedia)](https://en.wikipedia.org/wiki/Financial_market)
- [Portfolio (Finance) (Wikipedia)](https://en.wikipedia.org/wiki/Portfolio_(finance))

## License
This project is for educational purposes. Fork, modify, and experiment freely!
