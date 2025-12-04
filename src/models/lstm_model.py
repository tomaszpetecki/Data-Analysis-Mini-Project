import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model

#THIS was a good idea to play around with, and I often had to update some things, so I made it into a class
class LSTMModel:
    def __init__(self, time_steps: int, n_features: int, n_units=64, dropout_rate=0.2, model_path=None):
        """
        Args:
        - time_steps: Number of time steps (days) the model will look back to predict the next day.
        - n_features: Number of features in the input data (e.g., 9 features per day).
        - n_units: Number of units in the LSTM layer (default: 64).
        - dropout_rate: Dropout rate to prevent overfitting (default: 0.2).
        """
        self.time_steps = time_steps
        self.n_features = n_features
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.model = None
        if model_path:
            self.load_model(model_path)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.n_units, activation='relu', input_shape=(self.time_steps, self.n_features), return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification: 1 for migraine, 0 for no migraine
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train: np.array, y_train: np.array, epochs: int = 10, batch_size: int = 32, validation_data=None):
        if self.model is None:
            self.build_model()
        
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X_test: np.array, y_test: np.array):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

    def save_model(self, model_path: str):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict(self, X: np.array) -> np.array:
        return (self.model.predict(X) > 0.5).astype(int)  # Threshold at 0.5 for binary classification
    
