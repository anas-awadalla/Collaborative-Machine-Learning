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
        x1 = layers.Dense(64, activation="relu", name="layer0")(inputs)
        x2 = layers.Dense(64, activation="relu", name="layer1")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        # self.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def update_weights(self, avg_gradient):
        self.optimizer.apply_gradients(zip(avg_gradient, self.model.trainable_weights))
    
    def save_weights(self, path):
            tfjs.converters.save_keras_model(self.model, path)

    def get_weights(self):
        # TODO: return weights in a map where keys are layer names and values are arrays of weights
        weights = {}
        for layer in self.model.layers:
            # iterate through kernels and bias weights
            for weight in layer.weights:
                weights[weight.name] = weight.numpy().tolist()
        print(weights.keys())
        return weights