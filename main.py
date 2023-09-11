Here are the optimized versions of the functions in the Python script:

1. get_stock_data:
```python


def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


```

2. preprocess_data:
```python


def preprocess_data(stock_data, target_col):
    # Feature engineering
    stock_data['SMA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
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


```

3. calculate_rsi:
```python


def calculate_rsi(close_prices, window=14):
    diff = close_prices.diff(1).dropna()
    up = diff * 0
    down = up.copy()
    up[diff > 0] = diff[diff > 0]
    down[diff < 0] = -diff[diff < 0]
    up[up.index[window-1]] = np.mean(up[:window])
    up = up.drop(up.index[:(window-1)])
    down[down.index[window-1]] = np.mean(down[:window])
    down = down.drop(down.index[:(window-1)])
    rs = up.rolling(window).mean() / down.rolling(window).mean()
    rsi = 100 - (100 / (1 + rs))
    return rsi


```

4. calculate_macd:
```python


def calculate_macd(close_prices, n_fast=12, n_slow=26):
    ema_fast = close_prices.ewm(span=n_fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd


```

5. split_dataset:
```python


def split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test


```

6. train_random_forest:
```python


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model


```

7. train_lstm:
```python


def train_lstm(X_train, y_train, epochs, batch_size):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([
        LSTM(units=50, return_sequences=True,
             input_shape=(X_train.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=1, callbacks=[EarlyStopping(patience=3)])
    return model


```

8. generate_trading_signals:
```python


def generate_trading_signals(model, X_test):
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)
    signals = ['Buy' if predicted_prices[i] > predicted_prices[i-1] else 'Sell' if predicted_prices[i]
               < predicted_prices[i-1] else 'Hold' for i in range(1, len(predicted_prices))]
    return signals


```

9. perform_backtesting:
```python


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


```

10. automate_trading:
```python


def automate_trading(symbol, model):
    stock_data = get_stock_data(symbol, '2022-01-01', '2022-12-31')
    X, y = preprocess_data(stock_data, 'Close')
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    signals = generate_trading_signals(model, X_test)
    returns = perform_backtesting(y_test, signals, initial_investment=10000)
    print(f"Final balance: ${returns[-1]}")


```

11. display_web_interface:
```python


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


```

You can replace the respective functions in the original script with the updated versions for optimization.
