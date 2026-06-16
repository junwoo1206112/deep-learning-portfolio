import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class GRUPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build_model(self, input_shape: tuple):
        model = Sequential([
            GRU(self.config['units'], return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(self.config['units'] // 2),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, callbacks=None):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        return history.history

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
