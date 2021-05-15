import numpy as np

global g_Nt,g_K
g_Nt=64
g_K=4

def cre_hk(Nt,Np,K):        #Nt为天线数，Np为路径数,K为用户数               
    hk_real=np.zeros(shape=(1,Nt))
    hk_imag=np.zeros(shape=(1,Nt))
    for i in range(Np):
        at_real=np.array([])
        at_imag=np.array([])
        for x in range(Nt):
            aod=np.sin(np.random.uniform(1.0,360.0))
            at_real=np.append(at_real,np.cos(np.pi*x*aod)) 
            at_imag=np.append(at_imag,np.sin(np.pi*x*aod))         #天线距离为0.5倍波长
            # print(i,x,aod,at)
        alpha_real=np.random.normal(loc=0.0,scale=1.0)
        alpha_imag=np.random.normal(loc=0.0,scale=1.0)
        hk_real=hk_real+np.sqrt(Nt/Np)*alpha_real*at_real-1/np.sqrt(Np)*alpha_imag*at_imag
        hk_imag=hk_imag+np.sqrt(Nt/Np)*alpha_real*at_imag+1/np.sqrt(Np)*alpha_imag*at_real
        # hk_real=np.mat(hk_real)
        # hk_imag=np.mat(hk_imag)

    hk_real=hk_real.T
    hk_imag=hk_imag.T
    # print('最开始',hk_real.shape,hk_imag.shape)
    for m in range(K-1):
        hk_real_tem=np.zeros(shape=(1,Nt))
        hk_imag_tem=np.zeros(shape=(1,Nt))
        for i in range(Np):
            at_real=np.array([])
            at_imag=np.array([])
            for x in range(Nt):
                aod=np.sin(np.random.uniform(1.0,360.0))
                at_real=np.append(at_real,np.cos(np.pi*x*aod)) 
                at_imag=np.append(at_imag,np.sin(np.pi*x*aod))         #天线距离为0.5倍波长
                # print(i,x,aod,at)
            alpha_real=np.random.normal(loc=0.0,scale=0.5)
            alpha_imag=np.random.normal(loc=0.0,scale=0.5)
            hk_real_tem=hk_real_tem+np.sqrt(Nt/Np)*alpha_real*at_real-1/np.sqrt(Np)*alpha_imag*at_imag
            hk_imag_tem=hk_imag_tem+np.sqrt(Nt/Np)*alpha_real*at_imag+1/np.sqrt(Np)*alpha_imag*at_real
            # hk_real_tem=np.mat(hk_real_tem)
            # hk_imag_tem=np.mat(hk_imag_tem)

        hk_real_tem=hk_real_tem.T
        hk_imag_tem=hk_imag_tem.T
        hk_real=np.c_[hk_real,hk_real_tem]
        hk_imag=np.c_[hk_imag,hk_imag_tem]
        # print(m)
        # print(hk_real,hk_imag)
    return hk_real,hk_imag

def cre_V(freal,fimag,w1real,w1imag,w2real,w2imag,K):
    w4_real=np.linalg.inv(w2real+np.matmul(np.matmul(w2imag,np.linalg.inv(w2real)),w2imag))
    w4_imag=-1*np.matmul(np.linalg.inv(w2real+np.matmul(np.matmul(w2imag,np.linalg.inv(w2real)),w2imag)),np.matmul(w2imag,np.linalg.inv(w2real)))
    w_temreal=np.matmul(w1real,w4_real)-np.matmul(w1imag,w4_imag)
    w_temimag=np.matmul(w1real,w4_imag)+np.matmul(w1imag,w4_real)
    fw_real=np.matmul(freal,w_temreal)-np.matmul(fimag,w_temimag)
    fw_imag=np.matmul(freal,w_temimag)+np.matmul(fimag,w_temreal)
    w3_temreal=np.sum(fw_real*fw_real)+np.sum(fw_imag*fw_imag)
    w3_temimag=2*np.sum(fw_real*fw_imag)
    w3=np.sqrt(K/(np.complex(w3_temreal,w3_temimag)))
    W3_real=np.real(w3)
    W3_imag=np.imag(w3)
    V_real=W3_real*fw_real-W3_imag*fw_imag
    V_imag=W3_real*fw_imag+W3_imag*fw_real
    return V_real,V_imag

def rate_puser(hkreal,hkimag,v1real,v1imag,v2real,v2imag,v3real,v3imag,v4real,v4imag,seta=0.1):                  #计算单用户速率，论文（11）

    # print('hkreal=',hkreal.shape)
    rate=np.math.log10(1+( np.power(np.matmul(hkreal.T,v1real)-np.matmul(-hkimag.T,v1imag),2)
                          +np.power(np.matmul(hkreal.T,v1imag)+np.matmul(-hkimag.T,v1real),2)
                          )
                          /( np.power(np.matmul(hkreal.T,v2real)-np.matmul(-hkimag.T,v2imag),2)
                          +np.power(np.matmul(hkreal.T,v2imag)+np.matmul(-hkimag.T,v2real),2)
                          +np.power(np.matmul(hkreal.T,v3real)-np.matmul(-hkimag.T,v3imag),2)
                          +np.power(np.matmul(hkreal.T,v3imag)+np.matmul(-hkimag.T,v3real),2)
                          +np.power(np.matmul(hkreal.T,v4real)-np.matmul(-hkimag.T,v4imag),2)
                          +np.power(np.matmul(hkreal.T,v4imag)+np.matmul(-hkimag.T,v4real),2)
                          +seta*seta
                          )
                    )/np.math.log10(2.0)
    return rate

  
Hk_real,Hk_imag=cre_hk(g_Nt,6,g_K)
# print('real=',Hk_real.shape,'\n','imag=',Hk_imag.shape)
aod=np.arctan(Hk_imag/Hk_real)
# print('商=',Hk_imag/Hk_real,'\n','aod=',aod,aod.shape)

F_real=1/np.sqrt(g_Nt)*np.cos(aod)
F_imag=1/np.sqrt(g_Nt)*np.sin(aod)
# print('F_real',F_real.shape,'\n','F_imag',F_imag.shape)

Heq_real=np.matmul(Hk_real.T,F_real)-np.matmul(-Hk_imag.T,F_imag)
Heq_imag=np.matmul(Hk_real.T,F_imag)+np.matmul(-Hk_imag.T,F_real)

# print(Heq_real,'\n',Heq_imag)
W1_real=Heq_real.T
W1_imag=-Heq_imag.T

W2_real=np.matmul(Heq_real,W1_real)-np.matmul(Heq_imag,W1_imag)
W2_imag=np.matmul(Heq_real,W1_imag)+np.matmul(Heq_imag,W1_real)
w=W2_real*W2_imag
# print(W2_real,'\n',W2_imag,'\n',w)
v_real,v_imag=cre_V(F_real,F_imag,W1_real,W1_imag,W2_real,W2_imag,g_K)
# print(v_real,v_real.shape,'\n',v_imag,v_imag.shape)

rate1=rate_puser(Hk_real[:,0],Hk_imag[:,0],v_real[:,0],v_imag[:,0],v_real[:,1],v_imag[:,1],v_real[:,2],v_imag[:,2],v_real[:,3],v_imag[:,3])
rate2=rate_puser(Hk_real[:,1],Hk_imag[:,1],v_real[:,1],v_imag[:,1],v_real[:,0],v_imag[:,0],v_real[:,2],v_imag[:,2],v_real[:,3],v_imag[:,3])
rate3=rate_puser(Hk_real[:,2],Hk_imag[:,2],v_real[:,2],v_imag[:,2],v_real[:,1],v_imag[:,1],v_real[:,0],v_imag[:,0],v_real[:,3],v_imag[:,3])
rate4=rate_puser(Hk_real[:,3],Hk_imag[:,3],v_real[:,3],v_imag[:,3],v_real[:,1],v_imag[:,1],v_real[:,2],v_imag[:,2],v_real[:,0],v_imag[:,0])
sum_rate=rate1+rate2+rate3+rate4

print('sum_rate=',sum_rate)