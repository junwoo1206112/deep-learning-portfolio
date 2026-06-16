import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class AutoencoderDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.threshold = None

    def build_model(self, input_dim: int):
        encoding_dim = self.config['encoding_dim']

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoder = Dense(encoding_dim, activation='relu')(encoder)

        decoder = Dense(encoding_dim * 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)

        autoencoder = Model(input_layer, decoder)

        autoencoder.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse'
        )

        self.model = autoencoder
        return autoencoder

    def train(self, X_train, callbacks=None):
        if self.model is None:
            self.build_model(X_train.shape[1])

        history = self.model.fit(
            X_train, X_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        train_pred = self.model.predict(X_train, verbose=0)
        mse = np.mean(np.power(X_train - train_pred, 2), axis=1)
        self.threshold = np.percentile(mse, 95)

        return history.history

    def detect(self, X):
        pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - pred, 2), axis=1)
        anomalies = mse > self.threshold
        return mse

    def save(self, path: str):
        self.model.save(path)
        npy_path = path.replace('.keras', '_threshold.npy')
        np.save(npy_path, self.threshold)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
        npy_path = path.replace('.keras', '_threshold.npy')
        self.threshold = float(np.load(npy_path))
