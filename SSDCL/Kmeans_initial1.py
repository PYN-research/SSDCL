#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from wideresnet import WideResNet
import numpy as np
import torch 
from sklearn.cluster import KMeans
import scipy.stats
import scipy.misc
temperature=0.5
ndf=64
EPS = 10e-5
def cosine_similarity(inputs,mu):
    sim = np.matmul(inputs,mu.transpose(1,0))/temperature
    sim_softmax = np.exp(sim)/np.sum(np.exp(sim),1,keepdims=True)
    return sim_softmax
def kmeans_init(img, k):
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=4, max_iter=100, 
                              random_state=42)
    labels=kmeans_model.fit_predict(img)
    means = kmeans_model.cluster_centers_.T
    means=np.transpose(means,(1,0))
    means = np.array(means)
    cov = np.array([np.cov(img[labels == i].T) for i in range(k)], dtype=np.float32)
    ids = set(labels)
    pis = np.array([np.sum([labels == i]) / len(labels) for i in ids], dtype=np.float32)
    gamma = cosine_similarity(img,means)
    return means,pis,cov,gamma
def responsibilities(img,pis,mu,cov):
    identity=np.stack([np.eye(cov.shape[1]) for i in range(cov.shape[0])])
    cov=identity*cov
    cov_inv=np.linalg.inv(cov)
    pis=np.reshape(pis,(1,pis.shape[0]))
    img=np.expand_dims(img,axis=1)
    mu=np.expand_dims(mu,axis=0)
    res=np.transpose((img-mu),(1,0,2))
    exp=np.matmul(res,cov_inv)
    exp=np.transpose(np.multiply(exp,res),(1,0,2))
    exp=np.sum(exp,axis=2)
    sign,logdet=np.linalg.slogdet(cov)
    logdet=np.reshape(logdet,(1,logdet.shape[0]))
    exp=-0.5*(logdet+exp+(ndf*np.log(2*np.pi)))
    exp=pis*np.exp(exp)#(B//M,K)
    gamma=exp/(np.expand_dims(exp.sum(1),1))
    return gamma
def update_responsibility(img, means, cov, pis, k):
    responsibilities = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, mean=means[j], cov=cov[j]) for j in range(k)]).T
    # normalize for each row
    norm = np.sum(responsibilities, axis = 1)
    # convert to column vector
    norm = np.reshape(norm, (len(norm), 1))
    responsibilities = responsibilities / (norm+EPS)
    return responsibilities
def kmeans_merge(img,k,M,features_dim):
    img=img.data.cpu().numpy()
    Mu=[]
    R=[]
    C=[]
    P=[]
    for i in range(M):
        mu,pis,cov,responsibilities=kmeans_init(img[:,i,:],k)
        Mu.append(mu)
        R.append(responsibilities)
        C.append(cov)
        P.append(pis)
    Mu1=np.stack(Mu)
    Mu1=np.reshape(Mu1,(1,k,M,features_dim))
    R1=np.stack(R)
    Cov1=np.stack(C)
    Pis1=np.stack(P)
    return Mu1,Pis1,Cov1,R1

def KMeans_init(img,k,M,features_dim):
    center,pis,cov,response=kmeans_merge(img,k,M,features_dim)
    mu=torch.tensor(center,requires_grad=False)
    pis=torch.tensor(pis,requires_grad=False)
    pis=pis.transpose(1,0).unsqueeze(0)
    cov=torch.tensor(cov,requires_grad=False)
    cov=cov.transpose(1,0).unsqueeze(0)
    gamma=torch.tensor(response,requires_grad=False)                                                        
    gamma=gamma.transpose(1,0).transpose(2,1)
    s1=gamma.sum(0).unsqueeze(0)
    s1=torch.tensor(s1,requires_grad=False)
    s2=(gamma.unsqueeze(3)*img.unsqueeze(1)).sum(0).unsqueeze(0)
    s2=torch.tensor(s2,requires_grad=False)
    img1=img.unsqueeze(3)
    operator=np.matmul(img1,img1.transpose(3,2)).unsqueeze(1)
    s3=(gamma.unsqueeze(3).unsqueeze(4)*operator).sum(0).unsqueeze(0)
    state=mu,pis,cov,s1,s2,s3

    return state