import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.python.keras import initializers
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
        self.aod=self.add_weight(shape=(input_shape[-1]//2,g_la*g_K),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self,input):
        print('my_input=',input)
        w_real=tf.cos(self.aod)
        w_imag=tf.sin(self.aod)
        w_left=tf.concat([w_real,w_imag],0)
        w_right=tf.concat([-1*w_imag,w_real],0)
        w=tf.concat([w_left,w_right],1)
        return tf.matmul(input,w)
    
class myhide1(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        print('myhide1 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],2048),
                               initializer="random_normal",
                               trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        dtype=tf.float32,
                                        trainable=True)
        else:
            self.bias = None


    def call(self,input):
        return tf.nn.relu(tf.matmul(input, self.weight) + self.bias)

class myhide2(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        print('myhide2 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],1024),
                               initializer="random_normal",
                               trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        dtype=tf.float32,
                                        trainable=True)
        else:
            self.bias = None

    def call(self,input):
        return tf.nn.relu(tf.matmul(input, self.weight) + self.bias)

class myhide3(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        print('myhide3 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],512),
                               initializer="random_normal",
                               trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        dtype=tf.float32,
                                        trainable=True)
        else:
            self.bias = None

    def call(self,input):
        return tf.nn.relu(tf.matmul(input, self.weight) + self.bias)
 

class myhide4(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        print('myhide4 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],64),
                               initializer="random_normal",
                               trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        dtype=tf.float32,
                                        trainable=True)
        else:
            self.bias = None

    def call(self,input):
        return tf.matmul(input, self.weight) + self.bias


user_hk_real,user_hk_imag=mydata.cre_hk_real(6,64)
# print(user_hk_real.T,-user_hk_imag.T)

yk=mydata.cre_yk(64,999,user_hk_real,user_hk_imag,1)

yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)

Myinput=tf.keras.Input(shape=128,dtype=tf.float32)
# print('Myinput=',Myinput)
my_wrf=myWrf()
my_hide1=myhide1(2048)
my_hide2=myhide2(1024)
my_hide3=myhide3(512)
my_hide4=myhide4(64)

out=my_wrf(Myinput)
out=my_hide1(out)
out=my_hide2(out)
out=my_hide3(out)
out=my_hide4(out)

model=keras.Model(Myinput,out)
loss=-1*tf.math.log(1+g_P*(
                           tf.pow(tf.matmul(tf.convert_to_tensor(user_hk_real.T,dtype=tf.float32),tf.expand_dims(tf.math.cos(out),axis=-1))-tf.matmul(tf.convert_to_tensor(-user_hk_imag.T,dtype=tf.float32),tf.expand_dims(tf.math.sin(out),axis=-1)),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(user_hk_real.T,dtype=tf.float32),tf.expand_dims(tf.math.sin(out),axis=-1))+tf.matmul(tf.convert_to_tensor(-user_hk_imag.T,dtype=tf.float32),tf.expand_dims(tf.math.cos(out),axis=-1)),2)
                          )/(g_M*g_K*g_seta*g_seta)
                    )/tf.math.log(2.0)
model.add_loss(loss)
model.compile( optimizer = tf.keras.optimizers.SGD(lr = 0.0000001))
model.fit(yk,batch_size=2560,epochs=1000000)      #样本数199，batch_size=256貌似更好；样本数499，bz=2560，lr = 0.0000001




# def Wrf(K,La,M):
#     wrf_real=np.array([])
#     wrf_imag=np.array([])
#     for x in range(K*La):
#         wrf_real_tep=np.array([])
#         wrf_imag_tep=np.array([])
#         for i in range(M):
#             aod=np.random.uniform(1.0,360.0)
#             wrf_real_tep=np.append(wrf_real_tep,np.cos(aod))
#             wrf_imag_tep=np.append(wrf_imag_tep,np.sin(aod))
#         # print('real_tep=',wrf_real_tep)
#         # print('imag_tep=',wrf_imag_tep)

#         if x!=0:
#             wrf_real_tep=np.mat(wrf_real_tep)
#             wrf_imag_tep=np.mat(wrf_imag_tep)

#         wrf_real=np.r_[wrf_real,wrf_real_tep]
#         wrf_imag=np.r_[wrf_imag,wrf_imag_tep]
        
#         if x==0:
#             wrf_real=np.mat(wrf_real)
#             wrf_imag=np.mat(wrf_imag)
#     return wrf_real,wrf_imag



# def cre_yk(M,num,yhk_real,yhk_imag):
#     Wrf_real,Wrf_imag=Wrf(4,6,64)
#     tem_nk_real,tem_nk_imag=hk.nk(64)
#     yk_real=np.matmul(Wrf_real,yhk_real+tem_nk_real)-np.matmul(Wrf_imag,yhk_imag+tem_nk_imag)
#     yk_imag=np.matmul(Wrf_real,yhk_imag+tem_nk_imag)+np.matmul(Wrf_imag,yhk_real+tem_nk_real)
#     yk_real=yk_real.T
#     yk_imag=yk_imag.T
#     #print(yk_real,yk_imag)
#     for i in range(num):
#         Wrf_real,Wrf_imag=Wrf(4,6,64)
#         tem_nk_real,tem_nk_imag=hk.nk(64)
#         # print(i)
#         yk_real_tep=np.matmul(Wrf_real,yhk_real+tem_nk_real)-np.matmul(Wrf_imag,yhk_imag+tem_nk_imag)
#         yk_imag_tep=np.matmul(Wrf_real,yhk_imag+tem_nk_imag)+np.matmul(Wrf_imag,yhk_real+tem_nk_real)
#         yk_real_tep=yk_real_tep.T
#         yk_imag_tep=yk_imag_tep.T
#         #print(yk_real_tep,yk_imag_tep)
#         yk_real=np.c_[yk_real,yk_real_tep]
#         yk_imag=np.c_[yk_imag,yk_imag_tep]
#         #print('this time:',yk_real,yk_imag)
#     return yk_real,yk_imag

# user_hk_real,user_hk_imag=hk.cre_hk_real(6,64)
# yk_real,yk_imag=cre_yk(64,4,user_hk_real,user_hk_imag) #样本数=num+1
# yk_real=tf.convert_to_tensor(yk_real,dtype=tf.float32)
# yk_imag=tf.convert_to_tensor(yk_imag,dtype=tf.float32)
# # print('yk=',yk_real,'user_imag=',yk_imag)

# ykreal=tf.keras.Input(shape=(120,),dtype=tf.float32)
# ykimag=tf.keras.Input(shape=(120,),dtype=tf.float32)

# out=layers.Dense(2048,activation=tf.nn.relu)(ykreal)
# out=layers.Dense(1024,activation=tf.nn.relu)(out)
# out=layers.Dense(512,activation=tf.nn.relu)(out)
# out=layers.Dense(64,activation=None)(out)

# concat = keras.layers.concatenate([ykimag,out])

# output = keras.layers.Dense(1)(concat)
# output2 = keras.layers.Dense(1)(out)

# model = keras.models.Model(inputs = [ykreal,ykimag],outputs = [output,output2])
# # out=tf.math.exp(out)
# model.compile(loss="mean_squared_error", optimizer="adam")
# model.fit([yk_real,yk_imag], batch_size=120,epochs=10)


# model=keras.Model(yk_real,out)
# model2=keras.Model(ykimag,out)
# loss=-1*tf.math.log(1+P*math.pow(np.abs(tf.matmul(user_hk_real.H,out)),2)/(M*K*seta*seta))/tf.math.log(2.0)
# model.add_loss(loss)
# model.compile( optimizer = tf.keras.optimizers.SGD(lr = 0.001))
# model.fit(yk,batch_size=1000,epochs=50)
