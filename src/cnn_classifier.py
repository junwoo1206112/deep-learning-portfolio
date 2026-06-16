import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class CNNClassifier:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build_model(self, input_shape: tuple):
        filters = self.config.get('filters', [32, 64, 128])
        kernel_size = self.config.get('kernel_size', 3)

        layers = []
        for i, f in enumerate(filters):
            if i == 0:
                layers.append(Conv2D(f, (kernel_size, kernel_size), activation='relu', input_shape=input_shape))
            else:
                layers.append(Conv2D(f, (kernel_size, kernel_size), activation='relu'))
            layers.append(MaxPooling2D((2, 2)))

        layers += [
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ]

        model = Sequential(layers)

        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, callbacks=None):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2], 1))

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
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
