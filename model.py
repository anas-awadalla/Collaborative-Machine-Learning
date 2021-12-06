import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

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
        self.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def update_weights(self, avg_gradient):
        self.optimizer.apply_gradients(zip(avg_gradient, self.model.trainable_weights))
    
    def save_weights(self, path):
            tfjs.converters.save_keras_model(this.model, path)
