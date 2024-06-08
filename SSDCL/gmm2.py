#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch import nn
eps = torch.autograd.Variable(torch.FloatTensor([1.e-8]), requires_grad=False)

class GMM(nn.Module):
    """Implements a Gaussian Mixture Model."""
    def __init__(self, num_mixtures, dimension_embedding,n_rots,batch_size,momentum,mu,pis,cov,S1,S2,S3):
        """Creates a Gaussian Mixture Model.
        Args:
            num_mixtures (int): the number of mixtures the model should have.
            dimension_embedding (int): the number of dimension of the embedding
                space (can also be thought as the input dimension of the model)
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding
        self.n_rots=n_rots
        self.batch_size=batch_size
        self.momentum=momentum
        self.mixtures = Mixture(dimension_embedding,n_rots,num_mixtures,batch_size,momentum,mu,pis,cov,S1,S2,S3)

    def forward(self, inputs):
        inputs = torch.reshape(inputs, (self.n_rots, len(inputs)//self.n_rots, -1))
        inputs = inputs.transpose(1,0)
        out1,out2,out3,S1,S2,S3,Mu,Pis,Cov,gamma = self.mixtures(inputs,with_log=False)
        out1 = torch.diagonal(out1,dim1=2,dim2=3)      
        out1 = out1.mean(0)
        out1 = out1.unsqueeze(0)
        out3 = out3.sum(3)
        out3 = out3.mean(0)
        out3 = out3.unsqueeze(0)
        out = -out1+out2+out3
        out=out.sum()
        return out,S1,S2,S3,Mu,Pis,Cov,gamma

def cosine_similarity(x1 ,x2):
    eps = torch.autograd.Variable(torch.FloatTensor([1.e-8]), requires_grad=False)
    dot_prod = torch.matmul(x1,x2.transpose(2,1))
    dist_x1 = torch.norm(x1, p=2, dim=-1)
    dist_x2 = torch.norm(x2, p=2, dim=-1)
    return dot_prod / torch.max((dist_x1.unsqueeze(0)*dist_x2).unsqueeze(2), eps)

class Mixture(nn.Module):
    def __init__(self, dimension_embedding,n_rots,k,batch_size,momentum,mu,pis,cov,S1_old,S2_old,S3_old):
        super().__init__()
        self.dimension_embedding = dimension_embedding
        self.n_rots=n_rots
        self.batch_size=batch_size
        self.momentum=momentum
        self.k=k
        self.Phi = pis
        self.Phi = nn.Parameter(self.Phi, requires_grad=True)

        # Mu is the center/mean of the mixtures.
        self.mu = mu
        self.mu = nn.Parameter(self.mu, requires_grad=True)

        # Sigma encodes the shape of the gaussian 'bubble' of a given mixture.
        self.Sigma = cov
        #self.Sigma = torch.from_numpy(self.Sigma).float()
        self.Sigma = nn.Parameter(self.Sigma, requires_grad=True)
        
        #S1,S2, S3 is the subset of sufficient statistics
        self.S1 = S1_old
        #self.S1 = torch.from_numpy(self.S1).float()
        self.S1 = nn.Parameter(self.S1,requires_grad=False)
        
        self.S2 = S2_old
        #self.S2 = torch.from_numpy(self.S2).float()
        self.S2 = nn.Parameter(self.S2, requires_grad=False)
        
        self.S3 = S3_old
        #self.S2 = torch.from_numpy(self.S2).float()
        self.S3 = nn.Parameter(self.S3, requires_grad=False)
    def forward(self, samples, with_log=False):
        #E-step
        samples = samples.data.cpu()
        affiliations = self.e_step(samples, self.Phi, self.mu,self.Sigma)
        affiliations = affiliations.data.cpu()
        
        #M-step
        mu_up,pis_up,cov_up,S1,S2,S3 = self.m_step(samples,self.S1,self.S2,self.S3,affiliations)       
        identity = np.stack([np.eye(cov_up.shape[3]) for i in range(cov_up.shape[2])])
        identity = np.stack([identity for i in range(cov_up.shape[1])])
        identity = np.expand_dims(identity,axis=0)
        identity = torch.from_numpy(identity)
        cov=cov_up.float()*identity.float()
        Log=[]
        cov1 = cov.squeeze(0)
        for n in range(cov1.shape[0]):               
            log_det = [torch.slogdet(cov1[n][j])[1] for j in range(cov1.shape[1])]
            Log.append(log_det)
        Log_det = torch.FloatTensor(Log)
        Log_det = Log_det.unsqueeze(0).unsqueeze(3).data.cpu().numpy()
        cov_inv=torch.inverse(cov).cpu().float()
        #GMM-NCE-Loss 
        out_values = []
        for i in range(samples.shape[0]):
            mu=mu_up.unsqueeze(3).float()
            res = samples[i].unsqueeze(0).unsqueeze(1).unsqueeze(2)-mu
            res = res.float()
            diff = torch.matmul(res,cov_inv)*res
            diff = diff.sum(4)
            diff = diff.data.cpu().numpy()
            exp = -0.5*(Log_det+diff+(samples[i].shape[1]*np.log(np.pi)))
            exp = torch.from_numpy(exp)
            exp = torch.log(pis_up.unsqueeze(3))+exp
            exp1 = affiliations[i].unsqueeze(0).unsqueeze(3)*exp
            exp1 = exp1.sum(1)
            out_values.append(exp1.data.cpu().numpy())
        out_values = np.stack(out_values)
        out1 = torch.from_numpy(out_values).float()
        #add loss
        iden = np.stack([np.eye(out1.shape[2]) for i in range(out1.shape[0])])
        iden = np.expand_dims(iden,1)
        iden = torch.from_numpy(iden).float()
        out3 = out1-iden
        gamma_mean=affiliations.mean(0).unsqueeze(0)
        out2 = torch.log(gamma_mean)*gamma_mean
        out2 = out2.sum(1).unsqueeze(1)
        return out1,out2,out3,S1,S2,S3,mu_up,pis_up,cov,affiliations
    
    def e_step(self, samples,pis_old,mu_old,cov_old):
        # Updating assignments
        identity = np.stack([np.eye(cov_old.shape[3]) for i in range(cov_old.shape[2])])
        identity = np.stack([identity for i in range(cov_old.shape[1])])
        identity = np.expand_dims(identity,axis=0)
        identity = torch.from_numpy(identity).cpu().float()
        cov_old = cov_old.cpu().float()*identity
        cov=cov_old.squeeze(0)
        Log=[]
        for n in range(cov.shape[0]):
            log_det = [torch.slogdet(cov[n][j])[1] for j in range(cov.shape[1])]
            Log.append(log_det)
        Log_det = torch.FloatTensor(Log)
        Log_det = Log_det.unsqueeze(0).unsqueeze(3).data.cpu().numpy()
        cov_inv = torch.inverse(cov_old).cpu().float()
        out_values = []
        for i in range(samples.shape[0]):
            mu=mu_old.unsqueeze(3).float()
            res = samples[i].unsqueeze(0).unsqueeze(1).unsqueeze(2)-mu
            res = res.float()
            diff = torch.matmul(res,cov_inv)*res
            diff = diff.sum(4)
            diff = diff.data.cpu().numpy()
            exp = -0.5*(Log_det+diff+(samples[i].shape[1]*np.log(np.pi)))
            exp = torch.from_numpy(exp)
            exp = pis_old.unsqueeze(3)*exp
            exp = exp/(exp.sum(1).unsqueeze(1))
            exp1 = torch.diagonal(exp,dim1=2,dim2=3)
            exp1 = exp1.squeeze(0)
            out_values.append(exp1.data.cpu().numpy())
        out_values = np.stack(out_values)
        out1 = torch.from_numpy(out_values).float()
        return out1
    def m_step(self, samples,S1_old,S2_old,S3_old,affiliations):      
        s1 = affiliations.sum(0).unsqueeze(0)
        S1 = self.momentum*S1_old.detach().float()+(1-self.momentum)*s1
        # Updating S2
        s2 = torch.mul(affiliations.unsqueeze(3),samples.unsqueeze(1)).sum(0)
        S2 = self.momentum*S2_old.detach().float()+(1-self.momentum)*s2
        # Updating S3
        cross = torch.matmul(samples.unsqueeze(1).unsqueeze(4),torch.transpose(samples.unsqueeze(1).unsqueeze(4),3,4))
        s3 = affiliations.unsqueeze(3).unsqueeze(4).mul(cross).sum(0)
        S3 = self.momentum*S3_old.detach().float()+(1-self.momentum)*s3
        # Updating phi.
        phi = S1 / S1.sum(1).unsqueeze(1)
        Phi = phi.data

        # Updating mu.
        mu = S2 / (S1.unsqueeze(3))
        mu = torch.nn.functional.normalize(mu, p=2, dim=3, eps=1e-12)
        mu = mu.data

        # Updating Sigma.
        S11=S1.unsqueeze(3).unsqueeze(4)
        term1 = S3/(S11)
        term2 = torch.matmul(S2.unsqueeze(4),S2.unsqueeze(4).transpose(3,4)) / torch.mul(S11,S11)
        sigma = term1 -term2
        Sigma = sigma.data.float() 
        return mu,Phi,Sigma,S1,S2,S3
