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

# Define absolute path to store dataset
dataset_dir = "/Users/derr/Documents/CS3244/Project/3244StockMarket/resources"
os.makedirs(dataset_dir, exist_ok=True)  # Ensure the directory exists

# Download latest dataset version
dataset_path = os.path.join(dataset_dir, "price-volume-data-for-all-us-stocks-etfs")
print("Downloading dataset...")
path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
print("Path to dataset files:", path)

# List all .txt files
txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]

# Combine all .txt files into one DataFrame
df_list = []
for file in txt_files:
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)  # Read each file
    df_list.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(df_list, ignore_index=True)

# Save as a single CSV file
csv_path = os.path.join(path, "merged_stock_data.csv")
merged_df.to_csv(csv_path, index=False)

print(f"Merged CSV saved at: {csv_path}")

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
plt.plot(df.index, df['close'], label='Closing Price')
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Compute moving averages
df['MA_7'] = df['close'].rolling(window=7).mean()
df['MA_30'] = df['close'].rolling(window=30).mean()

# Drop rows with NaN values due to rolling window
df.dropna(inplace=True)

# Split dataset into training and testing
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['close', 'MA_7', 'MA_30']])
test_scaled = scaler.transform(test_data[['close', 'MA_7', 'MA_30']])

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
        # Forward pass
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict with LSTM
lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = lstm_model(X_test_tensor).cpu().numpy()

mse_lstm = mean_squared_error(y_test_seq, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
print(f'LSTM MSE: {mse_lstm:.4f}, MAE: {mae_lstm:.4f}')

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_seq, label='Actual')
plt.plot(y_pred_lstm, label='Predicted')
plt.title('LSTM Model Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.show()
