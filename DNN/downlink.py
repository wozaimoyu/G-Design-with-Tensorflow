import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np
import hk
import  os


class MyLayer(layers.Layer):
    def __init__(self, in_dim, out_dim):
        super(MyLayer, self).__init__()
        self.kernel = self.add_variable("w",(in_dim, out_dim),trainable=True)
        self.bias = self.add_variable("b",(out_dim,1),trainable=True)
    def call(self, input, training=None):
        output = tf.matmul(input, self.kernel) + self.bias
        return output

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_1 = MyLayer(64,2048)
        self.layer_2 = MyLayer(2048,1024)
        self.layer_3 = MyLayer(1024,512)
    def call(self, inputs, training=None):
        out=self.layer_1(inputs)
        out=tf.nn.relu(out)
        out=self.layer_2(out)
        out=tf.nn.relu(out)
        out=self.layer_3(out)
        return out

model=MyModel()


