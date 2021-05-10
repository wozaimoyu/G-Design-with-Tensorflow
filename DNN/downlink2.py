import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np
import hk
import os
import math
global g_M,g_K,g_P,g_seta
g_M=64
g_K=4
g_P=1
g_seta=0.1

def cre_yk(M,num,yhk_real,yhk_imag,pu):
    tem_nk_real,tem_nk_imag=hk.nk(64)
    yk_real=yhk_real+tem_nk_real
    yk_imag=yhk_imag+tem_nk_imag
    yk_real=yk_real.T
    yk_imag=yk_imag.T
    #print(yk_real,yk_imag)
    for i in range(num):
        tem_nk_real,tem_nk_imag=hk.nk(64)
        # print(i)
        yk_real_tep=yhk_real+tem_nk_real
        yk_imag_tep=yhk_imag+tem_nk_imag
        yk_real_tep=yk_real_tep.T
        yk_imag_tep=yk_imag_tep.T
        #print(yk_real_tep,yk_imag_tep)
        yk_real=np.c_[yk_real,yk_real_tep]
        yk_imag=np.c_[yk_imag,yk_imag_tep]
        #print('this time:',yk_real,yk_imag)
        yk=np.c_[yk_real,yk_imag]
    return yk.T

user_hk_real,user_hk_imag=hk.cre_hk_real(6,64)
# print(user_hk_real.T,-user_hk_imag.T)

yk=cre_yk(64,499,user_hk_real,user_hk_imag,1)

yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)

Myinput=tf.keras.Input(shape=1,dtype=tf.float32)

out=layers.Dense(48,activation=None)(Myinput)
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
model.fit(yk,batch_size=2560,epochs=10000)

