import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import math
import random


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
        alpha_real=np.random.normal(loc=0.0,scale=1.0)
        alpha_imag=np.random.normal(loc=0.0,scale=1.0)
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
        nk_real=np.append(nk_real,np.random.normal(loc=0.0,scale=0.1))
        nk_imag=np.append(nk_imag,np.random.normal(loc=0.0,scale=0.1))
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

# user_hk_real,user_hk_imag=cre_hk_real(6,64)
# # print(user_hk_real.T,-user_hk_imag.T)

# yk=cre_yk(64,499,user_hk_real,user_hk_imag,1)

# yk=tf.convert_to_tensor(yk,dtype=tf.float32)
# print(yk)
