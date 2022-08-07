import tensorflow as tf

class Network(tf.keras.Model):

  def __init__(self, hidden_units=0):
    super().__init__()

    self.list_of_layers = []

    if hidden_units != 0:
        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation=tf.nn.sigmoid)
        self.list_of_layers.append(self.hidden_layer)

    self.output_layer = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)
    self.list_of_layers.append(self.output_layer)

  def call(self, x):
    for layer in self.list_of_layers:
        x = layer(x)
    return x

