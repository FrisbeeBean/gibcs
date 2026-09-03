import torch
import torch.nn as nn
from args import modelargs
class RoPE(nn.Module):
    def __init__(self,dim,args:modelargs,theta_power=10000.0):
        super().__init__()
        self.dim=dim
        thetai=1.0/(theta_power**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer("thetai",thetai)
        self.max_seq_n=args.max_seq_n
        self._build_cache(args.max_seq_n)

    def _build_cache(self,seq_n):
        t=torch.arange(seq_n,dtype=self.thetai.dtype,device=self.thetai.device)
        te=t.unsqueeze(1)*self.thetai.unsqueeze(0)
        final=torch.cat((te,te),dim=-1)
        self.register_buffer("cos_cache",final.cos()[None,None,:,:])
        self.register_buffer("sin_cache",final.sin()[None,None,:,:])

    def forward(self,i,seq_n):
        if seq_n>self.max_seq_n:
            self._build_cache(seq_n)
            self.max_seq_n=seq_n
        return (self.cos_cache[:,:,:seq_n,:].to(i.dtype),self.sin_cache[:,:,:seq_n,:].to(i.dtype))
    
def rotary_embedding(q,k,cos,sin):
    """It is used to apply RoPE on query and key values"""
    def helper(ip):
        ip1=ip[...,:ip.shape[-1]//2]
        ip2=ip[...,ip.shape[-1]//2:]
        return torch.cat((-ip2,ip1),dim=-1)
    q_embed=(q*cos)+(helper(q)*sin)
    k_embed=(k*cos)+(helper(k)*sin)
    return q_embed,k_embed