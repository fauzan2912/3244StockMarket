import os
import IPython.display as display
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, request
import math

# Define absolute path to store dataset
dataset_dir = "/Users/derr/Documents/CS3244/Project/3244StockMarket/stock_market/Resources"
os.makedirs(dataset_dir, exist_ok=True)  # Ensure the directory exists

# Download latest dataset version
dataset_path = os.path.join(dataset_dir, "price-volume-data-for-all-us-stocks-etfs")
print("Downloading dataset...")
path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
print("Path to dataset files:", path)

# Recursively find all .txt files in the subdirectories
txt_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.txt'):
            txt_files.append(os.path.join(root, file))

# Check if any .txt files were found
print(f"Text files found: {txt_files}")

# Combine all .txt files into one DataFrame if files exist
if txt_files:
    df_list = []
    for file in txt_files:
        df = pd.read_csv(file)  # Read each file
        df_list.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save as a single CSV file
    csv_path = os.path.join(dataset_dir, "merged_stock_data.csv")
    merged_df.to_csv(csv_path, index=False)

    print(f"Merged CSV saved at: {csv_path}")
else:
    print("No .txt files found in the dataset.")

# Load the merged dataset
df = pd.read_csv(csv_path)

# Display first few rows
display.display(df.head())

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

# Drop rows with NaN values due to rolling window
df.dropna(inplace=True)

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
mae_lr = mean_absolute_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)
print(f'Linear Regression MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}')

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, 1:])  # Features (MA_7, MA_30)
        y.append(data[i+sequence_length, 0])     # Target (Close)
    return np.array(X), np.array(y)

sequence_length = 30
X_train_seq, y_train_seq = create_sequences(train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(test_scaled, sequence_length)

# Convert data to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1).to(device)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step's output
        out = self.fc(out)
        return out

# Initialize model, loss, and optimizer
input_size = X_train_seq.shape[2]
hidden_size = 50
output_size = 1
num_layers = 2

lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Train LSTM model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ARIMA Model
def arima_model(train, test):
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    return predictions

# Prepare data for ARIMA
train_arima = train_data['Close'].values
test_arima = test_data['Close'].values

# Fit ARIMA model
arima_predictions = arima_model(train_arima, test_arima)

# Evaluate ARIMA
mse_arima = mean_squared_error(test_arima, arima_predictions)
print(f'ARIMA MSE: {mse_arima:.4f}')

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    # Here you can add logic to fetch data for the specific stock symbol
    # For now, we'll use the existing data
    return render_template('results.html',
                           lr_pred=round(y_pred[-1], 2),
                           lstm_pred=round(lstm_model(X_test_tensor[-1].unsqueeze(0)).item(), 2),
                           arima_pred=round(arima_predictions[-1], 2))

if __name__ == '__main__':
    app.run(debug=True)