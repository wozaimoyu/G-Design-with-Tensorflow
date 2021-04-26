import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np
import hk
import os
import math
global M,K,P,seta
M=64
K=4
P=1
seta=0.1
    

def Wrf(K,La,M):
    wrf=[[np.exp(complex(0,np.random.uniform(1.0,360.0))) for i in range(1,M+1)] for j in range(1,K*La+1)]
    return wrf
def nk(M):
    nk=[[complex(np.random.normal(loc=0.0,scale=0.1),np.random.normal(loc=0.0,scale=0.1))] for i in range(1,M+1)]
    return nk
def cre_yk(M,num,hk):
    yk=np.array([],dtype=complex)
    for i in range(num):
        tem_nk=nk(M)
        if(i%1000==0):
            print(i)
        yk=np.append(yk,np.matmul(Wrf(4,6,M),hk+tem_nk)) 
    yk=np.mat(yk)
    return yk.T
user_hk=hk.h_k()
yk=cre_yk(M,2000,user_hk)
out=yk

out=layers.Dense(2048,activation=tf.nn.relu)(out)
out=layers.Dense(1024,activation=tf.nn.relu)(out)
out=layers.Dense(512,activation=tf.nn.relu)(out)
out=layers.Dense(64,activation=None)(out)
out=tf.math.exp(out)

model=keras.Model(yk,out)
loss=-1*tf.math.log(1+P*math.pow(np.abs(tf.matmul(user_hk.H,out)),2)/(M*K*seta*seta))/tf.math.log(2.0)
model.add_loss(loss)
model.compile( optimizer = tf.keras.optimizers.SGD(lr = 0.001))
model.fit(yk,batch_size=2400,epochs=50)




# class MyLayer(layers.Layer):
#     def __init__(self, in_dim, out_dim):
#         super(MyLayer, self).__init__()
#         self.kernel = self.add_variable("w",(in_dim, out_dim),trainable=True)
#         self.bias = self.add_variable("b",(out_dim,1),trainable=True)
#     def call(self, input, training=None):
#         output = tf.matmul(input, self.kernel) + self.bias
#         return output

# class MyModel(keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # self.layer_1 = MyLayer(Nrf*La,2048)
#         self.layer_2 = MyLayer(2048,1024)
#         self.layer_3 = MyLayer(1024,512)
#         self.layer_4 = MyLayer(512,64)
#     def call(self, x_train, training=None):
#         out=self.layer_1(inputs)
#         out=tf.nn.relu(out)
#         out=self.layer_2(out)
#         out=tf.nn.relu(out)
#         out=self.layer_3(out)
#         out=tf.nn.relu(out)
#         out=self.layer_4(out)
#         return out








