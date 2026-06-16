import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))

        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build_model(self, input_shape: tuple):
        d_model = self.config['d_model']
        num_heads = self.config['num_heads']
        dff = self.config.get('dff', d_model * 4)
        num_layers = self.config.get('num_layers', 2)
        dropout_rate = self.config.get('dropout_rate', 0.1)

        inputs = Input(shape=input_shape)
        x = Dense(d_model)(inputs)
        x = PositionalEncoding(d_model)(x)

        for _ in range(num_layers):
            x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

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
