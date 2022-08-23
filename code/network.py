import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import RandomUniform

class Network(tf.keras.Model):

  def __init__(self, hidden_units=0):
    super().__init__()

    initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None)

    self.list_of_layers = []

    if hidden_units != 0:
        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation=tf.nn.sigmoid, kernel_initializer = initializer)
        self.list_of_layers.append(self.hidden_layer)

    self.output_layer = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid, kernel_initializer = initializer)
    self.list_of_layers.append(self.output_layer)

    weights = np.empty(0)
    for layer in self.list_of_layers:
      np.append(weights, layer.get_weights())
    self.weights_shape = len(weights.flatten())

  def call(self, x):
    for layer in self.list_of_layers:
        x = layer(x)
    return x

  def test(self):
    test = self.build([192,1])
    for layer in self.list_of_layers:
      print("weights:", len(layer.weights))
      print("trainable_weights:", len(layer.trainable_weights))
      print("non_trainable_weights:", len(layer.non_trainable_weights))

