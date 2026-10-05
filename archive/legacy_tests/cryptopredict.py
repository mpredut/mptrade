import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
import time
import os
import threading

class CryptoPredictor:
    def __init__(self, symbol="BTC/USDT", timeframe="30m", look_back=60, model_path="crypto_model.h5", update_interval=60, predict_interval=6):
        """
        Constructorul clasei CryptoPredictor.
        :param symbol: Simbolul criptomonedei (ex. "BTC/USDT").
        :param timeframe: Time interval of the collected data (e.g. "20m" for 20 minutes).
        :param look_back: Number of historical minutes used for the prediction.
        :param model_path: Path of the file where the model is saved.
        :param update_interval: Interval (in seconds) at which the model is updated.
        :param predict_interval: Interval (in seconds) at which a prediction is made.
        """
        self.symbol = symbol
        self.timeframe = timeframe  # Change this to "20m" for 20-minute data.
        self.look_back = look_back
        self.model_path = model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.update_interval = update_interval
        self.predict_interval = predict_interval

    def fetch_data(self, limit=500):
        """ Collect the historical data from Binance """
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def preprocess_data(self, data):
        """ Normalise and shape the data for LSTM training """
        data_close = data[['close']].values
        scaled_data = self.scaler.fit_transform(data_close)
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self):
        """ Build the LSTM model with Dropout """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=50, batch_size=32):
        """ Train the model and save it at the end """
        data = self.fetch_data()
        X_train, y_train = self.preprocess_data(data)
        
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self.model.save(self.model_path)
        print("The model was trained and saved.")

    def load_model(self):
        """ Load the saved model for predictions """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("The model was loaded.")
        else:
            print("The model was not found. Train it first.")

    def predict_next(self):
        """ Price prediction for the next 20-minute period """
        data = self.fetch_data(limit=self.look_back)
        data_close = data[['close']].values
        scaled_data = self.scaler.transform(data_close)
        
        last_days = scaled_data[-self.look_back:]  # The last 60 minutes, used for the prediction.
        last_days = np.reshape(last_days, (1, self.look_back, 1))
        
        predicted_price = self.model.predict(last_days)
        predicted_price = self.scaler.inverse_transform(predicted_price)  # Undo the normalisation.
        return predicted_price[0][0]

    def start_update_service(self):
        """ Update the model periodically """
        while True:
            print("Actualizare model...")
            self.train_model(epochs=1, batch_size=32)  # Train periodically.
            time.sleep(self.update_interval)

    def start_prediction_service(self):
        """ Periodic prediction service """
        while True:
            predicted_price = self.predict_next()
            print(f"Estimated price for the next period: {predicted_price}")
            time.sleep(self.predict_interval)

# Crearea obiectului CryptoPredictor
predictor = CryptoPredictor()

# Load the saved model.
predictor.load_model()

# Run the periodic update service and the prediction service in parallel.
update_thread = threading.Thread(target=predictor.start_update_service, name="start_update_service")
predict_thread = threading.Thread(target=predictor.start_prediction_service, name="start_prediction_service")

update_thread.start()
time.sleep(60)
predict_thread.start()
