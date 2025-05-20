import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_fee_percent=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_value_history = []
        self.balance_history = []
        self.shares_history = []
        
        return self._get_observation()
    
    def step(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        self.current_step += 1
        
        # Get current price and handle potential issues
        current_price = self.data.iloc[self.current_step]['Close']
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        
        # Safety check to prevent division by zero or negative prices
        if current_price <= 0:
            current_price = 0.01  # Use a small positive value instead
        
        # Execute action
        if action == 1:  # Buy
            # Calculate maximum shares we can buy
            max_shares = int(self.balance // current_price)
            shares_to_buy = max_shares
            
            # Calculate transaction fee
            transaction_fee = shares_to_buy * current_price * self.transaction_fee_percent
            
            # Check if we can afford at least one share plus fee
            if shares_to_buy > 0 and self.balance >= (shares_to_buy * current_price + transaction_fee):
                # Update balance and shares
                self.balance -= (shares_to_buy * current_price + transaction_fee)
                self.shares_held += shares_to_buy
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Calculate transaction fee
                transaction_fee = self.shares_held * current_price * self.transaction_fee_percent
                
                # Update balance and shares
                self.balance += (self.shares_held * current_price - transaction_fee)
                
                self.total_shares_sold += self.shares_held
                self.total_sales_value += (self.shares_held * current_price)
                
                self.shares_held = 0
        
        # Calculate portfolio value
        portfolio_value = self.balance + self.shares_held * current_price
        self.total_value_history.append(portfolio_value)
        self.balance_history.append(self.balance)
        self.shares_history.append(self.shares_held)
        
        # Calculate reward (change in portfolio value)
        if len(self.total_value_history) > 1:
            reward = self.total_value_history[-1] - self.total_value_history[-2]
        else:
            reward = 0
        
        # Check if done (end of data)
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        observation = self._get_observation()
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        # If at the end of data, return the last observation
        if self.current_step >= len(self.data) - 1:
            return np.zeros(28, dtype=np.float32)  # Return zeros as a placeholder
        
        # Get price data for the observation window (5 days)
        start_idx = max(0, self.current_step - 4)  # 5 days including current
        end_idx = self.current_step + 1
        
        # Ensure we have 5 days of data by padding with zeros if needed
        padding_days = 5 - (end_idx - start_idx)
        
        # Extract features from price data
        features = []
        
        # Add padding if needed
        for _ in range(padding_days):
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add actual data
        for i in range(start_idx, end_idx):
            # Extract price data safely
            open_price = self.data.iloc[i]['Open']
            high_price = self.data.iloc[i]['High']
            low_price = self.data.iloc[i]['Low']
            close_price = self.data.iloc[i]['Close'] 
            volume = self.data.iloc[i]['Volume']
            
            # Convert to float if Series
            if isinstance(open_price, pd.Series): open_price = open_price.iloc[0]
            if isinstance(high_price, pd.Series): high_price = high_price.iloc[0]
            if isinstance(low_price, pd.Series): low_price = low_price.iloc[0]
            if isinstance(close_price, pd.Series): close_price = close_price.iloc[0]
            if isinstance(volume, pd.Series): volume = volume.iloc[0]
            
            features.extend([
                float(open_price),
                float(high_price),
                float(low_price),
                float(close_price),
                float(volume)
            ])
        
        # Add current portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        if isinstance(current_price, pd.Series): current_price = current_price.iloc[0]
        
        features.extend([
            float(self.balance),
            float(self.shares_held),
            float(self.shares_held * current_price)  # Current position value
        ])
        
        return np.array(features, dtype=np.float32)
    
    def render(self):
        profit = self.total_value_history[-1] - self.initial_balance
        
        # Get current price and ensure it's a scalar, not a Series
        current_price = self.data.iloc[self.current_step]['Close']
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
            
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Portfolio value: ${self.total_value_history[-1]:.2f}")
        print(f"Profit/Loss: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)")
        print("--------------------")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # Reduced memory size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1  # Increased minimum epsilon for more exploration
        self.epsilon_decay = 0.9  # Faster epsilon decay
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        # Simplified model architecture
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))  # Reduced neurons
        model.add(Dense(32, activation='relu'))  # Reduced neurons
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Predict next action values using target model
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        
        # Get current Q values from main model
        target_f = self.model.predict(states, verbose=0)
        
        # Update Q values for acted actions
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def preprocess_data(data):
    """Preprocess the stock data"""
    # Drop any NaN values
    data = data.dropna()
    
    # Create copies of the original values before normalization (for testing)
    data_orig = data.copy()
    
    # Normalize data
    scaler = MinMaxScaler()
    numerical_cols = ['Open', 'High', 'Low', 'Close']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Normalize volume separately
    volume_scaler = MinMaxScaler()
    data['Volume'] = volume_scaler.fit_transform(data[['Volume']])
    
    return data, scaler, volume_scaler, data_orig


def train_agent(ticker='AAPL', episodes=20, batch_size=16):  # Reduced episodes and batch size
    # Fetch data for a shorter period
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)  # 1 year of data instead of 2
    
    print(f"Fetching historical data for {ticker} from {start_date.date()} to {end_date.date()}")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Skip preprocessing for training - use the original data directly
    print("Using original (non-normalized) data for training...")
    
    # Split data into training and testing (use a larger portion for testing)
    train_data = data.iloc[:int(len(data)*0.7)]  # 70% for training
    test_data = data.iloc[int(len(data)*0.7):]   # 30% for testing
    
    print(f"Training data size: {len(train_data)} days")
    print(f"Testing data size: {len(test_data)} days")
    
    # Create environment
    env = TradingEnvironment(train_data)
    
    # Get state and action dimensions
    state = env.reset()
    state_size = len(state)
    action_size = 3  # hold, buy, sell
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Track portfolio values for plotting
    portfolio_values = []
    
    # Training loop
    print("\nStarting training...")
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Get action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                # Update target model every episode
                agent.update_target_model()
                
                # Print episode summary
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                print(f"Final Portfolio Value: ${env.total_value_history[-1]:.2f}")
                print(f"Return: {(env.total_value_history[-1] - env.initial_balance) / env.initial_balance * 100:.2f}%")
                print("----------------------------------------")
                
                portfolio_values.append(env.total_value_history[-1])
                break
            
            # Train the agent with a smaller batch size and less frequently
            if len(agent.memory) > batch_size and e % 2 == 0:  # Train every other episode
                agent.replay(batch_size)
    
    # Save the trained model
    model_filename = f"trading_dqn_{ticker}.weights.h5"
    agent.save(model_filename)
    print(f"Model saved as {model_filename}")
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values)
    plt.title(f'Portfolio Value Over Training Episodes ({ticker})')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(f'training_progress_{ticker}.png')
    plt.close()  # Close instead of show to avoid blocking
    
    return agent, test_data


def test_agent(agent, test_data, ticker='AAPL'):
    # Create testing environment
    env = TradingEnvironment(test_data)
    
    # Get first state
    state = env.reset()
    
    # Keep track of actions and portfolio value
    actions_taken = []
    portfolio_values = []
    
    # Run through test data
    done = False
    while not done:
        # Choose action
        action = agent.act(state, training=False)
        actions_taken.append(action)
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
        # Record portfolio value
        portfolio_values.append(env.total_value_history[-1])
        
        # Render environment occasionally
        if env.current_step % 30 == 0:  # Print every 30 days instead of 20
            env.render()
    
    # Final portfolio value and return
    final_value = env.total_value_history[-1]
    roi = (final_value - env.initial_balance) / env.initial_balance * 100
    
    print("====== Testing Results ======")
    print(f"Initial Balance: ${env.initial_balance:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Return on Investment: {roi:.2f}%")
    
    # Count actions
    action_counts = {
        'Hold': actions_taken.count(0),
        'Buy': actions_taken.count(1),
        'Sell': actions_taken.count(2)
    }
    print("Action Distribution:")
    for action, count in action_counts.items():
        print(f"{action}: {count} ({count/len(actions_taken)*100:.2f}%)")
    
    # Plot portfolio value over time
    plt.figure(figsize=(10, 6))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title(f'Portfolio Value During Testing ({ticker})')
    plt.ylabel('Portfolio Value ($)')
    
    # Plot stock price
    plt.subplot(2, 1, 2)
    plt.plot(test_data['Close'].values)
    plt.title(f'{ticker} Stock Price During Testing Period')
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price ($)')
    
    plt.tight_layout()
    plt.savefig(f'testing_results_{ticker}.png')
    plt.close()  # Close instead of show
    
    # Plot trade actions on price chart
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['Close'].values, label='Stock Price', color='blue', alpha=0.6)
    
    # Plot buy and sell points
    for i, action in enumerate(actions_taken):
        if action == 1:  # Buy
            plt.scatter(i, test_data['Close'].iloc[i], color='green', marker='^', alpha=0.7)
        elif action == 2:  # Sell
            plt.scatter(i, test_data['Close'].iloc[i], color='red', marker='v', alpha=0.7)
    
    plt.title(f'Trading Actions on {ticker} Stock')
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price ($)')
    plt.legend(['Stock Price', 'Buy', 'Sell'])
    plt.savefig(f'trading_actions_{ticker}.png')
    plt.show()  # Close instead of show
    
    return portfolio_values, actions_taken


def main():
    # Set the stock ticker
    ticker = 'AAPL'  # Can be changed to any valid ticker
    
    # Define model filename
    model_filename = f"trading_dqn_{ticker}.weights.h5"
    
    # Check if a trained model already exists
    import os
    
    if os.path.exists(model_filename):
        print(f"Found existing trained model: {model_filename}")
        print("Skipping training and loading the existing model...")
        
        # Fetch data to create test dataset
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        
        print(f"Fetching historical data for {ticker} from {start_date.date()} to {end_date.date()}")
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Split data into training and testing
        test_data = data.iloc[int(len(data)*0.7):]  # Use the same 30% split as in train_agent
        
        # Create agent with appropriate dimensions
        # We need to create an environment temporarily to get the state size
        temp_env = TradingEnvironment(test_data)
        state = temp_env.reset()
        state_size = len(state)
        action_size = 3  # hold, buy, sell
        
        # Create agent and load weights
        agent = DQNAgent(state_size, action_size)
        agent.load(model_filename)
        print("Model loaded successfully!")
        
    else:
        print(f"No existing model found at {model_filename}")
        print(f"Training trading agent for {ticker}...")
        try:
            agent, test_data = train_agent(ticker=ticker, episodes=20)  # Reduced episodes
        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print(f"\nTesting trading agent on unseen {ticker} data...")
    try:
        portfolio_values, actions_taken = test_agent(agent, test_data, ticker)
        
        print("\nTraining and testing complete!")
        print("Check the generated plots for visual analysis of the agent's performance.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()