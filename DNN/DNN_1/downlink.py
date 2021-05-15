from numpy import linalg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.python.keras import initializers
import numpy as np
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.numpy_ops.np_dtypes import set_allow_float64
import mydata_1
import os
import math


 

global g_M,g_K,g_P,g_seta,g_la,g_ld,hide1,hide2,hide3
g_M=64
g_K=4
g_P=1
g_seta=0.1
g_la=7
g_ld=3
hide1=2048
hide2=1024
hide3=512

def cre_v(Hreal,Himag,vrf,ld,seta=0.1):        #创建混合预编码矩阵V=VRF*VD
    vrf_real=np.cos(vrf)                       #vrf为模拟预编码矩阵VRF的相位
    vrf_imag=np.sin(vrf)
    Heq_real=np.matmul(vrf_real.T,Hreal)-np.matmul(-1*vrf_imag.T,Himag)       #生成论文（16）的Heq
    Heq_imag=np.matmul(vrf_real.T,Himag)+np.matmul(-1*vrf_imag.T,Hreal)

    N_real1,N_imag1=mydata_1.nk(g_M)
    N_real2,N_imag2=mydata_1.nk(g_M)
    N_real3,N_imag3=mydata_1.nk(g_M)
    N_real4,N_imag4=mydata_1.nk(g_M)                                             #四个用户的噪声矩阵
    N_real=np.c_[N_real1,N_real2,N_real3,N_real4]
    N_imag=np.c_[N_imag1,N_imag2,N_imag3,N_imag4]
    Z_real=np.matmul(vrf_real.T,N_real)-np.matmul(-1*vrf_imag.T,N_imag)        #论文（18）的Z矩阵
    Z_imag=np.matmul(vrf_real.T,N_imag)+np.matmul(-1*vrf_imag.T,N_real)
    Y_real=Heq_real+Z_real                                                     #论文（18）的Y矩阵
    Y_imag=Heq_imag+Z_imag
    H_eq_real=1/(ld+seta*seta)*Y_real
    H_eq_imag=1/(ld+seta*seta)*Y_imag                                          #论文（19）的H_eq矩阵
    for i in range(ld-1):
        N_real1,N_imag1=mydata_1.nk(g_M)
        N_real2,N_imag2=mydata_1.nk(g_M)
        N_real3,N_imag3=mydata_1.nk(g_M)
        N_real4,N_imag4=mydata_1.nk(g_M)
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

    
class myWrf(tf.keras.layers.Layer):
    def __init__(self,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )

    def build(self,input_shape):
        # print('input_shape=',input_shape)
        self.aod=self.add_weight(shape=(input_shape[-1]//2,g_la*g_K),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self,input):
        # print('my_input=',input)
        w_real=tf.cos(self.aod)
        w_imag=tf.sin(self.aod)
        w_left=tf.concat([w_real,w_imag],0)
        w_right=tf.concat([-1*w_imag,w_real],0)
        w=tf.concat([w_left,w_right],1)
        return tf.matmul(input,w)

class myhide12(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        # print('myhide3 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],4096),
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

class myhide1(tf.keras.layers.Layer):
    def __init__(self,units,bias_initializer='zeros',use_bias=True,dtype=tf.float32):
        super().__init__(
            dtype=dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self,input_shape):
        # print('myhide1 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],hide1),
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
        # print('myhide2 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],hide2),
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
        # print('myhide3 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],hide3),
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
        # print('myhide4 input_shape=',input_shape)
        self.weight=self.add_weight(shape=(input_shape[-1],g_M),
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


user_hk_real,user_hk_imag=mydata_1.cre_hk_real(6,g_M)
# print(user_hk_real.T,-user_hk_imag.T)

yk=mydata_1.cre_yk(g_M,999,user_hk_real,user_hk_imag,1)

yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)

Myinput=tf.keras.Input(shape=128,dtype=tf.float32)
# print('Myinput=',Myinput)
my_wrf=myWrf()              #初始化WRF层
my_hide1=myhide1(hide1)      #初始化四个隐藏层
my_hide2=myhide2(hide2)
my_hide3=myhide3(hide3)
# my_hide12=myhide12(4096)
my_hide4=myhide4(64)

out=my_wrf(Myinput)
# out=my_hide12(out)
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
model.fit(yk,batch_size=2560,epochs=10000)      #样本数199，batch_size=256貌似更好；样本数499，bz=2560，lr = 0.0000001

sum_rate1=0
sum_rate2=0
sum_rate3=0

for i in range(100000):
    if i%100==0:
        print('循环了',i,'次')
    vrf1=model(mydata_1.yk1).numpy().T           #生成K个用户的vrf
    # print('vrf1=',vrf1,vrf1.shape)

    # user_hk_real2,user_hk_imag2=mydata_1.cre_hk_real(6,g_M)
    # yk2=mydata_1.cre_yk(g_M,0,user_hk_real2,user_hk_imag2,1)
    # yk2=tf.convert_to_tensor(yk2,dtype=tf.float32)    
    vrf2=model(mydata_1.yk2).numpy().T
    # print('vrf2=',vrf2,vrf2.shape)

    # user_hk_real3,user_hk_imag3=mydata_1.cre_hk_real(6,g_M)
    # yk3=mydata_1.cre_yk(g_M,0,user_hk_real3,user_hk_imag3,1)
    # yk3=tf.convert_to_tensor(yk3,dtype=tf.float32)    
    vrf3=model(mydata_1.yk3).numpy().T
    # print('vrf3=',vrf3,vrf3.shape)

    # user_hk_real4,user_hk_imag4=mydata_1.cre_hk_real(6,g_M)
    # yk4=mydata_1.cre_yk(g_M,0,user_hk_real4,user_hk_imag4,1)
    # yk4=tf.convert_to_tensor(yk4,dtype=tf.float32)    
    vrf4=model(mydata_1.yk4).numpy().T
    # print('vrf4=',vrf4,vrf4.shape)

    vrf=np.c_[vrf1,vrf2,vrf3,vrf4]        #堆叠成VRF的相位
    # print('vrf=',vrf,vrf.shape)
    hreal=np.c_[mydata_1.user_hk_real1,mydata_1.user_hk_real2,mydata_1.user_hk_real3,mydata_1.user_hk_real4]     #信道矩阵H
    himag=np.c_[mydata_1.user_hk_imag1,mydata_1.user_hk_imag2,mydata_1.user_hk_imag3,mydata_1.user_hk_imag4]
    # print('hreal=',hreal,hreal.shape)
    # print('himag=',himag,himag.shape)


    Vreal1,Vimag1=cre_v(hreal,himag,vrf,3)         #生成ld=3的时候的V，计算用户总速率
    rate11=rate_puser(mydata_1.user_hk_real1,mydata_1.user_hk_imag1,Vreal1[:,0],Vimag1[:,0],Vreal1[:,1],Vimag1[:,1],Vreal1[:,2],Vimag1[:,2],Vreal1[:,3],Vimag1[:,3])
    rate12=rate_puser(mydata_1.user_hk_real2,mydata_1.user_hk_imag2,Vreal1[:,1],Vimag1[:,1],Vreal1[:,0],Vimag1[:,0],Vreal1[:,2],Vimag1[:,2],Vreal1[:,3],Vimag1[:,3])
    rate13=rate_puser(mydata_1.user_hk_real3,mydata_1.user_hk_imag3,Vreal1[:,2],Vimag1[:,2],Vreal1[:,1],Vimag1[:,1],Vreal1[:,0],Vimag1[:,0],Vreal1[:,3],Vimag1[:,3])
    rate14=rate_puser(mydata_1.user_hk_real4,mydata_1.user_hk_imag4,Vreal1[:,3],Vimag1[:,3],Vreal1[:,0],Vimag1[:,0],Vreal1[:,2],Vimag1[:,2],Vreal1[:,1],Vimag1[:,1])
    sum_rate_tem1=rate11+rate12+rate13+rate14  

    if sum_rate_tem1>sum_rate1:
        sum_rate1=sum_rate_tem1
        vrf_aod_saved1=vrf
        h_real_saved1=hreal
        h_imag_saved1=himag
        Vreal_saved1=Vreal1
        Vimag_saved1=Vimag1

    Vreal2,Vimag2=cre_v(hreal,himag,vrf,2)      #生成ld=2的时候的V，计算用户总速率
    rate21=rate_puser(mydata_1.user_hk_real1,mydata_1.user_hk_imag1,Vreal2[:,0],Vimag2[:,0],Vreal2[:,1],Vimag2[:,1],Vreal2[:,2],Vimag2[:,2],Vreal2[:,3],Vimag2[:,3])
    rate22=rate_puser(mydata_1.user_hk_real2,mydata_1.user_hk_imag2,Vreal2[:,1],Vimag2[:,1],Vreal2[:,0],Vimag2[:,0],Vreal2[:,2],Vimag2[:,2],Vreal2[:,3],Vimag2[:,3])
    rate23=rate_puser(mydata_1.user_hk_real3,mydata_1.user_hk_imag3,Vreal2[:,2],Vimag2[:,2],Vreal2[:,1],Vimag2[:,1],Vreal2[:,0],Vimag2[:,0],Vreal2[:,3],Vimag2[:,3])
    rate24=rate_puser(mydata_1.user_hk_real4,mydata_1.user_hk_imag4,Vreal2[:,3],Vimag2[:,3],Vreal2[:,0],Vimag2[:,0],Vreal2[:,2],Vimag2[:,2],Vreal2[:,1],Vimag2[:,1])
    sum_rate_tem2=rate21+rate22+rate23+rate24
    if sum_rate_tem2>sum_rate2:
        sum_rate2=sum_rate_tem2
        vrf_aod_saved2=vrf
        h_real_saved2=hreal
        h_imag_saved2=himag
        Vreal_saved2=Vreal2
        Vimag_saved2=Vimag2

    Vreal3,Vimag3=cre_v(hreal,himag,vrf,1)     #生成ld=1的时候的V，计算用户总速率
    rate31=rate_puser(mydata_1.user_hk_real1,mydata_1.user_hk_imag1,Vreal3[:,0],Vimag3[:,0],Vreal3[:,1],Vimag3[:,1],Vreal3[:,2],Vimag3[:,2],Vreal3[:,3],Vimag3[:,3])
    rate32=rate_puser(mydata_1.user_hk_real2,mydata_1.user_hk_imag2,Vreal3[:,1],Vimag3[:,1],Vreal3[:,0],Vimag3[:,0],Vreal3[:,2],Vimag3[:,2],Vreal3[:,3],Vimag3[:,3])
    rate33=rate_puser(mydata_1.user_hk_real3,mydata_1.user_hk_imag3,Vreal3[:,2],Vimag3[:,2],Vreal3[:,1],Vimag3[:,1],Vreal3[:,0],Vimag3[:,0],Vreal3[:,3],Vimag3[:,3])
    rate34=rate_puser(mydata_1.user_hk_real4,mydata_1.user_hk_imag4,Vreal3[:,3],Vimag3[:,3],Vreal3[:,0],Vimag3[:,0],Vreal3[:,2],Vimag3[:,2],Vreal3[:,1],Vimag3[:,1])
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

    if i%1000==0:
        print('第',i,'次循环')
        print('sum_rate1=',sum_rate1)
        print('sum_rate2=',sum_rate2)
        print('sum_rate3=',sum_rate3)


# print('rate1=',rate1)
# print('rate2=',rate2)
# print('rate3=',rate3)
# print('rate3=',rate3)

print('sum_rate1=',sum_rate1,'vrf_aod1=',vrf_aod_saved1,
       'hreal1=',h_real_saved1,'himag1=',h_imag_saved1,'vreal1=',Vreal1,'vimag1=',Vimag1,'\n'
      )

print('sum_rate2=',sum_rate2,'vrf_aod2=',vrf_aod_saved2,
       'hreal2=',h_real_saved2,'himag2=',h_imag_saved2,'vreal2=',Vreal2,'vimag1=',Vimag2,'\n'
       )

print('sum_rate3=',sum_rate3,'vrf_aod3=',vrf_aod_saved3,
       'hreal3=',h_real_saved3,'himag2=',h_imag_saved3,'vreal3=',Vreal3,'vimag1=',Vimag3,'\n'
       )

print('sum_rate1=',sum_rate1)
print('sum_rate2=',sum_rate2)
print('sum_rate3=',sum_rate3)

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
