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
        inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1')(inputs)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10, activation='softmax', name='dense')(x)
        self.model = keras.Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=[['categorical_crossentropy'],['accuracy']])
        self.optimizer = keras.optimizers.Adam(learning_rate=5e-4)


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
        return weights

    def test_model(self):
        _ , (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        return [round(score[0], 3), round(score[2], 3)*100]
