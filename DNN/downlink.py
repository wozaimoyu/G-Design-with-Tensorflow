import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np
import  os


class SAELayer(layers.Layer):
    def __init__(self, num_outputs):
        super(SAELayer, self).__init__()
        self.num_outputs = num_outputs
	
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        self.bias = self.add_variable("bias",shape=[self.num_outputs])
    def call(self, input):
        output = tf.matmul(input, self.kernel) + self.bias
        output = tf.nn.relu(output)
        return output
class SAEModel(Model):
    def __init__(self, input_shape, output_shape, hidden_shape=None):
        if hidden_shape == None:
        hidden_shape = 0.5 * input_shape
        super(SAEModel, self).__init__()
        self.layer_1 = SAELayer(hidden_shape)
        self.layer_2 = layers.Dense(output_shape, activation=tf.nn.relu)


    def call(self, input_tensor, training=False):
        hidden = self.layer_1(input_tensor)
        output = self.layer_2(hidden)
        return output
    def loss()
        loss=-np.log2(1+1/2.56*abs())
        


