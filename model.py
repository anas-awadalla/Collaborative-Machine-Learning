import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs
import numpy as np
class mnist_model():
    """
    Model for MNIST
    """
    def __init__(self) -> None:
        inputs = keras.Input(shape=(784,), name="digits")
        x1 = layers.Dense(512, activation="relu", name="layer0")(inputs)
        x2 = layers.Dense(256, activation="relu", name="layer1")(x1)
        x3 = layers.Dense(128, activation="relu", name="layer2")(x2)
        x4 = layers.Dense(64, activation="relu", name="layer3")(x3)
        x5 = layers.Dense(32, activation="relu", name="layer4")(x4)
        outputs = layers.Dense(10, activation="softmax", name="predictions")(x5)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def update_weights(self, avg_gradient):
        self.optimizer.apply_gradients(zip(avg_gradient, self.model.trainable_weights))
    
    def save_weights(self, path):
            tfjs.converters.save_keras_model(self.model, path)

    def get_weights(self):
        weights = {}
        for layer in self.model.layers:
            # iterate through kernels and bias weights
            for weight in layer.weights:
                weights[weight.name[:-2]] = weight.numpy().tolist()
        # print(weights.keys())
        return weights
