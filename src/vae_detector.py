import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam


class VAELossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        original, reconstruction, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(original - reconstruction), axis=-1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        self.add_loss(reconstruction_loss + kl_loss)
        return reconstruction


class VAEDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.threshold = None

    def build_model(self, input_dim: int):
        encoding_dim = self.config['encoding_dim']
        latent_dim = self.config.get('latent_dim', 8)

        inputs = Input(shape=(input_dim,))
        h = Dense(encoding_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        decoder_h = Dense(encoding_dim, activation='relu')
        decoder_mean = Dense(input_dim)

        h_decoded = decoder_h(z)
        outputs = decoder_mean(h_decoded)

        vae_outputs = VAELossLayer()([inputs, outputs, z_mean, z_log_var])
        vae = Model(inputs, vae_outputs)

        vae.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=None
        )

        self.model = vae
        return vae

    def train(self, X_train, callbacks=None):
        if self.model is None:
            self.build_model(X_train.shape[1])

        history = self.model.fit(
            X_train,
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
        return mse

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
