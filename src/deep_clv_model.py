import tensorflow as tf
import numpy as np
import joblib

class DeepCLVModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y, epochs=50, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        y_pred_log = self.model.predict(np.array(X)).flatten()
        y_pred = np.expm1(y_pred_log)
        return np.maximum(0, y_pred)

    def save(self, path):
        if not path.endswith('.keras'):
            path = path + '.keras'
        self.model.save(path)

    @staticmethod
    def load(path):
        if not path.endswith('.keras'):
            path = path + '.keras'
        return DeepCLVModel(model_path=path)
