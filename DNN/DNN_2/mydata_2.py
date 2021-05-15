import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import math
import random

global g_M,g_K,g_P,g_seta,g_la,g_ld
g_M=64
g_K=4
g_P=1
g_seta=0.1
g_la=6
g_ld=2

def cre_hk_real(lp,M):
    hk_real=np.zeros(shape=(1,M))
    hk_imag=np.zeros(shape=(1,M))
    for i in range(lp):
        at_real=np.array([])
        at_imag=np.array([])
        for x in range(M):
            aod=np.sin(np.random.uniform(1.0,360.0))
            at_real=np.append(at_real,np.cos(np.pi*x*aod)) 
            at_imag=np.append(at_imag,np.sin(np.pi*x*aod))         #天线距离为0.5倍波长
            # print(i,x,aod,at)
        alpha_real=np.random.normal(loc=0.0,scale=0.5)
        alpha_imag=np.random.normal(loc=0.0,scale=0.5)
        hk_real=hk_real+1/np.sqrt(lp)*alpha_real*at_real-1/np.sqrt(lp)*alpha_imag*at_imag
        hk_imag=hk_imag+1/np.sqrt(lp)*alpha_real*at_imag+1/np.sqrt(lp)*alpha_imag*at_real
        hk_real=np.mat(hk_real)
        hk_imag=np.mat(hk_imag)
    # hk_real=tf.convert_to_tensor(hk_real.T,dtype=tf.float32)
    # hk_imag=tf.convert_to_tensor(hk_imag.T,dtype=tf.float32)
    # print('hk_real=',hk_real)
    # print('hk_imag=',hk_imag)
    # h_k=np.mat(hk)
        # print("hk等于",h_k)
    return hk_real.T,hk_imag.T
# a,b=cre_hk_real(6,6)
# print('a=',a)
# print('b=',b)

def nk(M):
    nk_real=np.array([])
    nk_imag=np.array([])
    for i in range(M):
        nk_real=np.append(nk_real,np.random.normal(loc=0.0,scale=0.05))
        nk_imag=np.append(nk_imag,np.random.normal(loc=0.0,scale=0.05))
    nk_real=np.mat(nk_real).T
    nk_imag=np.mat(nk_imag).T
    return nk_real,nk_imag

def cre_yk(M,num,yhk_real,yhk_imag,pu):
    tem_nk_real,tem_nk_imag=nk(64)
    yk_real=yhk_real+tem_nk_real
    yk_imag=yhk_imag+tem_nk_imag
    yk_real=yk_real.T
    yk_imag=yk_imag.T
    yk=np.c_[yk_real,yk_imag]      #维度(128,1)
    # yk=tf.convert_to_tensor(yk,dtype=tf.float32)
    # print('pre yk =',yk)
    for i in range(num):
        tem_nk_real,tem_nk_imag=nk(64)
        # print(i)
        yk_real_tep=yhk_real+tem_nk_real
        yk_imag_tep=yhk_imag+tem_nk_imag
        yk_real_tep=yk_real_tep.T
        yk_imag_tep=yk_imag_tep.T
        yk_tep=np.c_[yk_real_tep,yk_imag_tep]
        #print(yk_real_tep,yk_imag_tep)
        # yk_real=np.c_[yk_real,yk_real_tep]
        # yk_imag=np.c_[yk_imag,yk_imag_tep]
        #print('this time:',yk_real,yk_imag)
        yk=np.r_[yk,yk_tep]
        # yk=tf.convert_to_tensor(yk,dtype=tf.float32)
        # print(i,'yk=',yk)
    return yk     #维度(num+1,128)

def set_GPU():
    """GPU相关设置"""

    # 打印变量在那个设备上
    # tf.debugging.set_log_device_placement(True)
    # 获取物理GPU个数
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('物理GPU个数为：', len(gpus))
    # 设置内存自增长
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print('-------------已设置完GPU内存自增长--------------')

    # 设置哪个GPU对设备可见，即指定用哪个GPU
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))

set_GPU()

user_hk_real1,user_hk_imag1=cre_hk_real(6,g_M)
yk1=cre_yk(g_M,0,user_hk_real1,user_hk_imag1,1)
yk1=tf.convert_to_tensor(yk1,dtype=tf.float32)    #生成一个（1，128）的矩阵

user_hk_real2,user_hk_imag2=cre_hk_real(6,g_M)
yk2=cre_yk(g_M,0,user_hk_real2,user_hk_imag2,1)
yk2=tf.convert_to_tensor(yk2,dtype=tf.float32)   

user_hk_real3,user_hk_imag3=cre_hk_real(6,g_M)
yk3=cre_yk(g_M,0,user_hk_real3,user_hk_imag3,1)
yk3=tf.convert_to_tensor(yk3,dtype=tf.float32)   

user_hk_real4,user_hk_imag4=cre_hk_real(6,g_M)
yk4=cre_yk(g_M,0,user_hk_real4,user_hk_imag4,1)
yk4=tf.convert_to_tensor(yk4,dtype=tf.float32)
# user_hk_real,user_hk_imag=cre_hk_real(6,64)
# # print(user_hk_real.T,-user_hk_imag.T)

# yk=cre_yk(64,499,user_hk_real,user_hk_imag,1)

# yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)
