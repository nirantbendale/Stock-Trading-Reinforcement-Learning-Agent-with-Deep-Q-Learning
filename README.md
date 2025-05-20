# Deep Q-Learning Stock Trading Algorithm

A reinforcement learning approach to automated stock trading using Deep Q-Networks (DQN).

## Project Overview

This project implements an autonomous trading agent that learns to make optimal buy, sell, or hold decisions based on historical stock data. The agent uses Deep Q-Learning to develop strategies that aim to maximize returns while managing transaction costs.

## Technical Details

### Trading Environment
- Simulates realistic market interactions with 0.1% transaction fees
- Portfolio tracking with balance and position management
- Observation space: 28-dimensional state vector including 5-day price history and current portfolio status
- Action space: 3 possible actions (hold, buy, sell)

### DQN Agent
- Neural network architecture: 2 hidden layers with 32 neurons each
- Experience replay memory with 1000 past interactions
- Epsilon-greedy exploration strategy with decay from 1.0 to 0.1
- Discount factor (gamma): 0.95 for future reward valuation

### Training Parameters
- Training data: 1 year of historical stock prices
- 20 training episodes for reasonable convergence
- Batch size of 16 for network updates
- 70/30 train/test data split

## Implementation

The system fetches stock data from Yahoo Finance API, processes it through the trading environment, and trains the agent to make profitable decisions. The agent learns through trial and error, developing strategies based on price patterns and its own trading history.



## Requirements

- numpy
- pandas
- matplotlib
- tensorflow
- yfinance
- scikit-learn

## Results

The project generates visualizations showing:
- Portfolio value progression during training
- Comparison between portfolio performance and underlying stock performance
- Trading decisions (buy/sell points) overlaid on stock price charts
