import tensorflow as tf

class Network(tf.keras.Model):

  def __init__(self, hidden_units=0):
    super().__init__()

    self.layers = []

    if hidden_units != 0:
        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation=tf.nn.sigmoid)
        self.layers.append(self.hidden_layer)

    self.output_layer = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)
    self.layers.append(self.output_layer)

  def call(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

