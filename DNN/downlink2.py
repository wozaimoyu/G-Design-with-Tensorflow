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
g_la=7


class myWrf(tf.keras.layers.Layer):
    def __init__(self,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )

    def build(self,input_shape):
        # print('input_shape=',input_shape)
        self.aod=self.add_weight(
            shape=(input_shape[-1]//2,g_la*g_K),initializer="random_normal",trainable=True
        )

    def call(self,input):
        # print('my_input=',input)
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
model.fit(yk,batch_size=2560,epochs=1000)      #样本数199，batch_size=256貌似更好；样本数499，bz=2560，lr = 0.0000001



def cre_v(Hreal,Himag,vrf,ld,seta=0.1):        #创建混合预编码矩阵V=VRF*VD
    vrf_real=np.cos(vrf)                       #vrf为模拟预编码矩阵VRF的相位
    vrf_imag=np.sin(vrf)
    Heq_real=np.matmul(vrf_real.T,Hreal)-np.matmul(-1*vrf_imag.T,Himag)       #生成论文（16）的Heq
    Heq_imag=np.matmul(vrf_real.T,Himag)+np.matmul(-1*vrf_imag.T,Hreal)

    N_real1,N_imag1=mydata.nk(g_M)
    N_real2,N_imag2=mydata.nk(g_M)
    N_real3,N_imag3=mydata.nk(g_M)
    N_real4,N_imag4=mydata.nk(g_M)                                             #四个用户的噪声矩阵
    N_real=np.c_[N_real1,N_real2,N_real3,N_real4]
    N_imag=np.c_[N_imag1,N_imag2,N_imag3,N_imag4]
    Z_real=np.matmul(vrf_real.T,N_real)-np.matmul(-1*vrf_imag.T,N_imag)        #论文（18）的Z矩阵
    Z_imag=np.matmul(vrf_real.T,N_imag)+np.matmul(-1*vrf_imag.T,N_real)
    Y_real=Heq_real+Z_real                                                     #论文（18）的Y矩阵
    Y_imag=Heq_imag+Z_imag
    H_eq_real=1/(ld+seta*seta)*Y_real
    H_eq_imag=1/(ld+seta*seta)*Y_imag                                          #论文（19）的H_eq矩阵
    for i in range(ld-1):
        N_real1,N_imag1=mydata.nk(g_M)
        N_real2,N_imag2=mydata.nk(g_M)
        N_real3,N_imag3=mydata.nk(g_M)
        N_real4,N_imag4=mydata.nk(g_M)
        N_real=np.c_[N_real1,N_real2,N_real3,N_real4]
        N_imag=np.c_[N_imag1,N_imag2,N_imag3,N_imag4]
        Z_real=np.matmul(vrf_real.T,N_real)-np.matmul(-1*vrf_imag.T,N_imag)
        Z_imag=np.matmul(vrf_real.T,N_imag)+np.matmul(-1*vrf_imag.T,N_real)
        Y_real=Heq_real+Z_real
        Y_imag=Heq_imag+Z_imag
        H_eq_real_t=1/(ld+seta*seta)*Y_real
        H_eq_imag_t=1/(ld+seta*seta)*Y_imag
        H_eq_real=H_eq_real+H_eq_real_t
        H_eq_imag=H_eq_imag+H_eq_imag_t                                        #ld个Y相加得到最终的H_eq
    VD1_real=np.matmul(H_eq_real.T,H_eq_real)-np.matmul(-1*H_eq_imag.T,H_eq_imag)
    VD1_imag=np.matmul(H_eq_real.T,H_eq_imag)+np.matmul(-1*H_eq_imag.T,H_eq_real)       #见论文（20）
    VD2_real=np.linalg.inv(VD1_real+np.matmul(np.matmul(VD1_imag,np.linalg.inv(VD1_real)),VD1_imag))
    VD2_imag=-1*np.matmul(np.linalg.inv(VD1_real+np.matmul(np.matmul(VD1_imag,np.linalg.inv(VD1_real)),VD1_imag)),np.matmul(VD1_imag,np.linalg.inv(VD1_real)))
    VD_real=np.matmul(H_eq_real,VD2_real)-np.matmul(H_eq_imag,VD2_imag)
    VD_imag=np.matmul(H_eq_real,VD2_imag)+np.matmul(H_eq_imag,VD2_real)

    V_real=np.matmul(vrf_real,VD_real)-np.matmul(vrf_imag,VD_imag)
    V_imag=np.matmul(vrf_real,VD_imag)+np.matmul(vrf_imag,VD_real)
    return V_real,V_imag

def rate_puser(hkreal,hkimag,v1real,v1imag,v2real,v2imag,v3real,v3imag,v4real,v4imag,seta=0.1):                  #计算单用户速率，论文（11）
    v1real=tf.convert_to_tensor(v1real,dtype=tf.float32)
    v1imag=tf.convert_to_tensor(v1imag,dtype=tf.float32)
    v2real=tf.convert_to_tensor(v2real,dtype=tf.float32)
    v2imag=tf.convert_to_tensor(v2imag,dtype=tf.float32)
    v3real=tf.convert_to_tensor(v3real,dtype=tf.float32)
    v3imag=tf.convert_to_tensor(v3imag,dtype=tf.float32)
    v4real=tf.convert_to_tensor(v4real,dtype=tf.float32)
    v4imag=tf.convert_to_tensor(v4imag,dtype=tf.float32)
    # print('hkreal=',hkreal.shape)
    rate=tf.math.log(1+(   tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v1real)-tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v1imag),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v1imag)+tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v1real),2)
                          )
                          /( tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v2real)-tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v2imag),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v2imag)+tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v2real),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v3real)-tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v3imag),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v3imag)+tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v3real),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v4real)-tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v4imag),2)
                          +tf.pow(tf.matmul(tf.convert_to_tensor(hkreal.T,dtype=tf.float32),v4imag)+tf.matmul(tf.convert_to_tensor(-hkimag.T,dtype=tf.float32),v4real),2)
                          +seta*seta
                          )
                    )/tf.math.log(2.0)
    return rate

sum_rate1=0
sum_rate2=0
sum_rate3=0

for i in range(100000):
    if i%100==0:
        print('循环了',i,'次')
    vrf1=model(mydata.yk1).numpy().T           #生成K个用户的vrf
    # print('vrf1=',vrf1,vrf1.shape)

    # user_hk_real2,user_hk_imag2=mydata.cre_hk_real(6,g_M)
    # yk2=mydata.cre_yk(g_M,0,user_hk_real2,user_hk_imag2,1)
    # yk2=tf.convert_to_tensor(yk2,dtype=tf.float32)    
    vrf2=model(mydata.yk2).numpy().T
    # print('vrf2=',vrf2,vrf2.shape)

    # user_hk_real3,user_hk_imag3=mydata.cre_hk_real(6,g_M)
    # yk3=mydata.cre_yk(g_M,0,user_hk_real3,user_hk_imag3,1)
    # yk3=tf.convert_to_tensor(yk3,dtype=tf.float32)    
    vrf3=model(mydata.yk3).numpy().T
    # print('vrf3=',vrf3,vrf3.shape)

    # user_hk_real4,user_hk_imag4=mydata.cre_hk_real(6,g_M)
    # yk4=mydata.cre_yk(g_M,0,user_hk_real4,user_hk_imag4,1)
    # yk4=tf.convert_to_tensor(yk4,dtype=tf.float32)    
    vrf4=model(mydata.yk4).numpy().T
    # print('vrf4=',vrf4,vrf4.shape)

    vrf=np.c_[vrf1,vrf2,vrf3,vrf4]        #堆叠成VRF的相位
    # print('vrf=',vrf,vrf.shape)
    hreal=np.c_[mydata.user_hk_real1,mydata.user_hk_real2,mydata.user_hk_real3,mydata.user_hk_real4]     #信道矩阵H
    himag=np.c_[mydata.user_hk_imag1,mydata.user_hk_imag2,mydata.user_hk_imag3,mydata.user_hk_imag4]
    # print('hreal=',hreal,hreal.shape)
    # print('himag=',himag,himag.shape)


    Vreal1,Vimag1=cre_v(hreal,himag,vrf,3)         #生成ld=3的时候的V，计算用户总速率
    rate11=rate_puser(mydata.user_hk_real1,mydata.user_hk_imag1,Vreal1[:,0],Vimag1[:,0],Vreal1[:,1],Vimag1[:,1],Vreal1[:,2],Vimag1[:,2],Vreal1[:,3],Vimag1[:,3])
    rate12=rate_puser(mydata.user_hk_real2,mydata.user_hk_imag2,Vreal1[:,1],Vimag1[:,1],Vreal1[:,0],Vimag1[:,0],Vreal1[:,2],Vimag1[:,2],Vreal1[:,3],Vimag1[:,3])
    rate13=rate_puser(mydata.user_hk_real3,mydata.user_hk_imag3,Vreal1[:,2],Vimag1[:,2],Vreal1[:,1],Vimag1[:,1],Vreal1[:,0],Vimag1[:,0],Vreal1[:,3],Vimag1[:,3])
    rate14=rate_puser(mydata.user_hk_real4,mydata.user_hk_imag4,Vreal1[:,3],Vimag1[:,3],Vreal1[:,0],Vimag1[:,0],Vreal1[:,2],Vimag1[:,2],Vreal1[:,1],Vimag1[:,1])
    sum_rate_tem1=rate11+rate12+rate13+rate14  

    if sum_rate_tem1>sum_rate1:
        sum_rate1=sum_rate_tem1
        vrf_aod_saved1=vrf
        h_real_saved1=hreal
        h_imag_saved1=himag
        Vreal_saved1=Vreal1
        Vimag_saved1=Vimag1

    Vreal2,Vimag2=cre_v(hreal,himag,vrf,2)      #生成ld=2的时候的V，计算用户总速率
    rate21=rate_puser(mydata.user_hk_real1,mydata.user_hk_imag1,Vreal2[:,0],Vimag2[:,0],Vreal2[:,1],Vimag2[:,1],Vreal2[:,2],Vimag2[:,2],Vreal2[:,3],Vimag2[:,3])
    rate22=rate_puser(mydata.user_hk_real2,mydata.user_hk_imag2,Vreal2[:,1],Vimag2[:,1],Vreal2[:,0],Vimag2[:,0],Vreal2[:,2],Vimag2[:,2],Vreal2[:,3],Vimag2[:,3])
    rate23=rate_puser(mydata.user_hk_real3,mydata.user_hk_imag3,Vreal2[:,2],Vimag2[:,2],Vreal2[:,1],Vimag2[:,1],Vreal2[:,0],Vimag2[:,0],Vreal2[:,3],Vimag2[:,3])
    rate24=rate_puser(mydata.user_hk_real4,mydata.user_hk_imag4,Vreal2[:,3],Vimag2[:,3],Vreal2[:,0],Vimag2[:,0],Vreal2[:,2],Vimag2[:,2],Vreal2[:,1],Vimag2[:,1])
    sum_rate_tem2=rate21+rate22+rate23+rate24
    if sum_rate_tem2>sum_rate2:
        sum_rate2=sum_rate_tem2
        vrf_aod_saved2=vrf
        h_real_saved2=hreal
        h_imag_saved2=himag
        Vreal_saved2=Vreal2
        Vimag_saved2=Vimag2

    Vreal3,Vimag3=cre_v(hreal,himag,vrf,1)     #生成ld=1的时候的V，计算用户总速率
    rate31=rate_puser(mydata.user_hk_real1,mydata.user_hk_imag1,Vreal3[:,0],Vimag3[:,0],Vreal3[:,1],Vimag3[:,1],Vreal3[:,2],Vimag3[:,2],Vreal3[:,3],Vimag3[:,3])
    rate32=rate_puser(mydata.user_hk_real2,mydata.user_hk_imag2,Vreal3[:,1],Vimag3[:,1],Vreal3[:,0],Vimag3[:,0],Vreal3[:,2],Vimag3[:,2],Vreal3[:,3],Vimag3[:,3])
    rate33=rate_puser(mydata.user_hk_real3,mydata.user_hk_imag3,Vreal3[:,2],Vimag3[:,2],Vreal3[:,1],Vimag3[:,1],Vreal3[:,0],Vimag3[:,0],Vreal3[:,3],Vimag3[:,3])
    rate34=rate_puser(mydata.user_hk_real4,mydata.user_hk_imag4,Vreal3[:,3],Vimag3[:,3],Vreal3[:,0],Vimag3[:,0],Vreal3[:,2],Vimag3[:,2],Vreal3[:,1],Vimag3[:,1])
    sum_rate_tem3=rate31+rate32+rate33+rate34
    if sum_rate_tem3>sum_rate3:
        sum_rate3=sum_rate_tem3
        vrf_aod_saved3=vrf
        h_real_saved3=hreal
        h_imag_saved3=himag
        Vreal_saved3=Vreal3
        Vimag_saved3=Vimag3
    if sum_rate1>sum_rate2 and sum_rate2>sum_rate3:
        print('第',i,'次循环')
        print('sum_rate1=',sum_rate1)
        print('sum_rate2=',sum_rate2)
        print('sum_rate3=',sum_rate3)

# print('rate1=',rate1)
# print('rate2=',rate2)
# print('rate3=',rate3)
# print('rate3=',rate3)
print('sum_rate1=',sum_rate1)
print('sum_rate2=',sum_rate2)
print('sum_rate3=',sum_rate3)

