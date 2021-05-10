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
    
def Wrf(K,La,M):
    wrf_real=np.array([])
    wrf_imag=np.array([])
    for x in range(K*La):
        wrf_real_tep=np.array([])
        wrf_imag_tep=np.array([])
        for i in range(M):
            aod=np.random.uniform(1.0,360.0)
            wrf_real_tep=np.append(wrf_real_tep,np.cos(aod))
            wrf_imag_tep=np.append(wrf_imag_tep,np.sin(aod))
        # print('real_tep=',wrf_real_tep)
        # print('imag_tep=',wrf_imag_tep)

        if x!=0:
            wrf_real_tep=np.mat(wrf_real_tep)
            wrf_imag_tep=np.mat(wrf_imag_tep)

        wrf_real=np.r_[wrf_real,wrf_real_tep]
        wrf_imag=np.r_[wrf_imag,wrf_imag_tep]
        
        if x==0:
            wrf_real=np.mat(wrf_real)
            wrf_imag=np.mat(wrf_imag)
    return wrf_real,wrf_imag



def cre_yk(M,num,yhk_real,yhk_imag):
    Wrf_real,Wrf_imag=Wrf(4,6,64)
    tem_nk_real,tem_nk_imag=hk.nk(64)
    yk_real=np.matmul(Wrf_real,yhk_real+tem_nk_real)-np.matmul(Wrf_imag,yhk_imag+tem_nk_imag)
    yk_imag=np.matmul(Wrf_real,yhk_imag+tem_nk_imag)+np.matmul(Wrf_imag,yhk_real+tem_nk_real)
    yk_real=yk_real.T
    yk_imag=yk_imag.T
    #print(yk_real,yk_imag)
    for i in range(num):
        Wrf_real,Wrf_imag=Wrf(4,6,64)
        tem_nk_real,tem_nk_imag=hk.nk(64)
        # print(i)
        yk_real_tep=np.matmul(Wrf_real,yhk_real+tem_nk_real)-np.matmul(Wrf_imag,yhk_imag+tem_nk_imag)
        yk_imag_tep=np.matmul(Wrf_real,yhk_imag+tem_nk_imag)+np.matmul(Wrf_imag,yhk_real+tem_nk_real)
        yk_real_tep=yk_real_tep.T
        yk_imag_tep=yk_imag_tep.T
        #print(yk_real_tep,yk_imag_tep)
        yk_real=np.c_[yk_real,yk_real_tep]
        yk_imag=np.c_[yk_imag,yk_imag_tep]
        #print('this time:',yk_real,yk_imag)
    return yk_real,yk_imag

user_hk_real,user_hk_imag=hk.cre_hk_real(6,64)
yk_real,yk_imag=cre_yk(64,4,user_hk_real,user_hk_imag) #样本数=num+1
yk_real=tf.convert_to_tensor(yk_real,dtype=tf.float32)
yk_imag=tf.convert_to_tensor(yk_imag,dtype=tf.float32)
# print('yk=',yk_real,'user_imag=',yk_imag)

ykreal=tf.keras.Input(shape=(120,),dtype=tf.float32)
ykimag=tf.keras.Input(shape=(120,),dtype=tf.float32)

out=layers.Dense(2048,activation=tf.nn.relu)(ykreal)
out=layers.Dense(1024,activation=tf.nn.relu)(out)
out=layers.Dense(512,activation=tf.nn.relu)(out)
out=layers.Dense(64,activation=None)(out)

concat = keras.layers.concatenate([ykimag,out])

output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(out)

model = keras.models.Model(inputs = [ykreal,ykimag],outputs = [output,output2])
# out=tf.math.exp(out)
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit([yk_real,yk_imag], batch_size=120,epochs=10)


# model=keras.Model(yk_real,out)
# model2=keras.Model(ykimag,out)
# loss=-1*tf.math.log(1+P*math.pow(np.abs(tf.matmul(user_hk_real.H,out)),2)/(M*K*seta*seta))/tf.math.log(2.0)
# model.add_loss(loss)
# model.compile( optimizer = tf.keras.optimizers.SGD(lr = 0.001))
# model.fit(yk,batch_size=1000,epochs=50)










