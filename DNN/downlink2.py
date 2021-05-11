import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np
import mydata
import os
import math
global g_M,g_K,g_P,g_seta,g_la
g_M=64
g_K=4
g_P=1
g_seta=0.1
g_la=6


class myWrf(tf.keras.layers.Layer):
    def __init__(self,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )

    def build(self,input_shape):
        print('input_shape=',input_shape)
        self.aod=self.add_weight(
            shape=(input_shape[-1]//2,g_la*g_K),initializer="random_normal",trainable=True
        )

    def call(self,input):
        print('my_input=',input)
        w_real=tf.cos(self.aod)
        w_imag=tf.sin(self.aod)
        w_left=tf.concat([w_real,w_imag],0)
        w_right=tf.concat([-1*w_imag,w_real],0)
        w=tf.concat([w_left,w_right],1)
        return tf.matmul(input,w)


user_hk_real,user_hk_imag=mydata.cre_hk_real(6,64)
# print(user_hk_real.T,-user_hk_imag.T)

yk=mydata.cre_yk(64,999,user_hk_real,user_hk_imag,1)

yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)

Myinput=tf.keras.Input(shape=128,dtype=tf.float32)
# print('Myinput=',Myinput)
my_wrf=myWrf()

out=my_wrf(Myinput)
out=layers.Dense(2048,activation=tf.nn.relu)(out)
out=layers.Dense(1024,activation=tf.nn.relu)(out)
out=layers.Dense(512,activation=tf.nn.relu)(out)
out=layers.Dense(64,activation=None)(out)
# print('out=',out,'changed out=',tf.expand_dims(tf.math.cos(out),axis=-1))
# out_real=tf.math.cos(out)
# out_imag=tf.math.sin(out)

model=keras.Model(Myinput,out)
loss=-1*tf.math.log(1+g_P*(
                           tf.pow(tf.matmul(tf.convert_to_tensor(user_hk_real.T,dtype=tf.float32),tf.expand_dims(tf.math.cos(out),axis=-1))-tf.matmul(tf.convert_to_tensor(-user_hk_imag.T,dtype=tf.float32),tf.expand_dims(tf.math.sin(out),axis=-1)),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(user_hk_real.T,dtype=tf.float32),tf.expand_dims(tf.math.sin(out),axis=-1))+tf.matmul(tf.convert_to_tensor(-user_hk_imag.T,dtype=tf.float32),tf.expand_dims(tf.math.cos(out),axis=-1)),2)
                          )/(g_M*g_K*g_seta*g_seta)
                    )/tf.math.log(2.0)
model.add_loss(loss)
model.compile( optimizer = tf.keras.optimizers.SGD(lr = 0.0000001))
model.fit(yk,batch_size=2560,epochs=1000000)      #样本数199，batch_size=256貌似更好；样本数499，bz=2560，lr = 0.0000001

