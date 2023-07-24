import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

file_path = 'CLNX.MC.csv' # change file with historical Data
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

scaled_df = pd.DataFrame(scaled_data, columns=features)
scaled_df['Date'] = data['Date']

# Define the time steps for creating sequences
time_steps = 10

# Function to create sequences
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training
X_train, y_train = create_sequences(scaled_df[features], scaled_df['Close'], time_steps)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[early_stopping])

# Extend the dataset for future prediction
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=365, freq='D')[1:]
future_df = pd.DataFrame(index=future_dates, columns=features)
future_data = pd.concat([scaled_df, future_df], ignore_index=True)

# Scale the future data
X_future = scaler.transform(future_data[features])

# Perform predictions for the next 1 year (365 days)
X_pred = []
for i in range(len(X_future) - time_steps):
    X_pred.append(X_future[i:(i + time_steps)])
X_pred = np.array(X_pred)

y_pred = model.predict(X_pred)
y_pred_extended = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]

# Trim y_pred_extended to match future_dates
y_pred_extended_trimmed = y_pred_extended[:len(future_dates)]

# Plotting actual and predicted stock prices for the extended period (from today onwards)
plt.figure(figsize=(12, 6))
plt.plot(future_dates, y_pred_extended_trimmed, label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Predicted Prices for Stocks from Today Onwards')
plt.legend()
plt.show()
