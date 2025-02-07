import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Define dataset path
path = "/Users/derr/.cache/kagglehub/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/versions/3"

# List all .txt files in the directory
txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
print("Available .txt files:", txt_files)

# Ensure there are .txt files available
if not txt_files:
    raise FileNotFoundError("No .txt files found in the dataset directory.")

# Select the first .txt file (change index if needed)
file_name = txt_files[0]  # Change index if needed
full_path = os.path.join(path, file_name)

# Load the .txt file as a CSV
df = pd.read_csv(full_path, delimiter=",")  # Using "," since it follows CSV format
print("First few rows of the dataset:")
print(df.head())

# Display first few rows
display(df.head())

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Fill missing values with forward fill
df.fillna(method='ffill', inplace=True)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot stock closing price over time
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Closing Price')
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Compute moving averages
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

# Split dataset into training and testing
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['Close', 'MA_7', 'MA_30']])
test_scaled = scaler.transform(test_data[['Close', 'MA_7', 'MA_30']])

# Prepare features and target for Linear Regression
X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)
print(f'Linear Regression MSE: {mse_lr:.4f}')

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 1:])  # Features (MA_7, MA_30)
        y.append(data[i+seq_length, 0])     # Target (Close)
    return np.array(X), np.array(y)

seq_length = 30
X_train_seq, y_train_seq = create_sequences(train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(test_scaled, seq_length)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32)

# Predict with LSTM
y_pred_lstm = lstm_model.predict(X_test_seq)
mse_lstm = mean_squared_error(y_test_seq, y_pred_lstm)
print(f'LSTM MSE: {mse_lstm:.4f}')

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_seq, label='Actual')
plt.plot(y_pred_lstm, label='Predicted')
plt.title('LSTM Model Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.show()