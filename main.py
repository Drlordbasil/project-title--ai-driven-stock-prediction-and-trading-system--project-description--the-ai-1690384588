import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import matplotlib.pyplot as plt

# Function to collect stock data from Yahoo Finance API


def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess the stock data


def preprocess_data(stock_data, target_col):
    # Feature engineering
    stock_data['SMA'] = stock_data['Close'].rolling(
        window=20).mean()     # Simple Moving Average
    # Relative Strength Index (RSI)
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    # Moving Average Convergence Divergence (MACD)
    stock_data['MACD'] = calculate_macd(stock_data['Close'])

    # Drop missing values and unnecessary columns
    stock_data.dropna(inplace=True)
    stock_data.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)

    # Splitting the data into features and target variable
    X = stock_data.drop(target_col, axis=1)
    y = stock_data[target_col]

    # Scale the features using Min-Max scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to calculate Relative Strength Index (RSI)


def calculate_rsi(close_prices, window=14):
    diff = close_prices.diff(1).dropna()
    up = diff * 0
    down = up.copy()
    up[diff > 0] = diff[diff > 0]
    down[diff < 0] = -diff[diff < 0]
    # first value is average of gains
    up[up.index[window-1]] = np.mean(up[:window])
    up = up.drop(up.index[:(window-1)])
    # first value is average of losses
    down[down.index[window-1]] = np.mean(down[:window])
    down = down.drop(down.index[:(window-1)])
    rs = up.rolling(window).mean() / down.rolling(window).mean()
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Moving Average Convergence Divergence (MACD)


def calculate_macd(close_prices, n_fast=12, n_slow=26):
    ema_fast = close_prices.ewm(span=n_fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd

# Function to split the dataset into train and test sets


def split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

# Function to create and train a Random Forest Regressor model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model

# Function to create and train an LSTM model


def train_lstm(X_train, y_train, epochs, batch_size):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=1, callbacks=[EarlyStopping(patience=3)])
    return model

# Function to generate trading signals based on predicted price movements


def generate_trading_signals(model, X_test):
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)
    signals = []
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > predicted_prices[i-1]:
            signals.append('Buy')
        elif predicted_prices[i] < predicted_prices[i-1]:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Function to perform backtesting and evaluate trading strategies


def perform_backtesting(y_test, signals, initial_investment):
    returns = []
    position = 0
    balance = initial_investment
    for i in range(len(signals)):
        if signals[i] == 'Buy' and position == 0:
            position += balance / y_test[i]
            balance = 0
        elif signals[i] == 'Sell' and position > 0:
            balance += position * y_test[i]
            position = 0
        returns.append(balance)
    return returns

# Function to automate trading based on generated signals


def automate_trading(symbol, model):
    stock_data = get_stock_data(symbol, '2022-01-01', '2022-12-31')
    X, y = preprocess_data(stock_data, 'Close')
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    signals = generate_trading_signals(model, X_test)
    returns = perform_backtesting(y_test, signals, initial_investment=10000)
    print(f"Final balance: ${returns[-1]}")

# Function to display stock predictions, trading signals, portfolio performance, and other analytics


def display_web_interface(symbol, model):
    stock_data = get_stock_data(symbol, '2023-01-01', '2023-12-31')
    X, y = preprocess_data(stock_data, 'Close')
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    signals = generate_trading_signals(model, X_test)
    returns = perform_backtesting(y_test, signals, initial_investment=10000)

    # Plotting stock predictions and actual prices
    predicted_prices = model.predict(X_test)
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Plotting portfolio returns
    plt.plot(returns)
    plt.xlabel('Time')
    plt.ylabel('Portfolio Balance')
    plt.show()

# Main function


def main():
    # Stock symbol for prediction
    symbol = 'AAPL'

    # Collecting stock data for training and testing
    stock_data = get_stock_data(symbol, '2010-01-01', '2021-12-31')
    X, y = preprocess_data(stock_data, 'Close')
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

    # Training models
    random_forest_model = train_random_forest(X_train, y_train)
    lstm_model = train_lstm(X_train, y_train, epochs=100, batch_size=32)

    # Testing models and generating trading signals
    random_forest_signals = generate_trading_signals(
        random_forest_model, X_test)
    lstm_signals = generate_trading_signals(lstm_model, X_test)

    # Performing backtesting and evaluating trading strategies
    random_forest_returns = perform_backtesting(
        y_test, random_forest_signals, initial_investment=10000)
    lstm_returns = perform_backtesting(
        y_test, lstm_signals, initial_investment=10000)

    # Automating trading using the best model
    if random_forest_returns[-1] > lstm_returns[-1]:
        automate_trading(symbol, random_forest_model)
    else:
        automate_trading(symbol, lstm_model)

    # Displaying web interface for stock predictions, trading signals, and portfolio performance
    display_web_interface(symbol, lstm_model)  # Change model as required


if __name__ == '__main__':
    main()
