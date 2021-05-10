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


    
