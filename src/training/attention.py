import torch
import torch.nn as nn
import math
from args import modelargs
from rope import rotary_embedding
from cache import KVcache
def rpt_kv(hs:torch.Tensor,nr:int)->torch.Tensor:
    """Repeat key and value states for MQA"""
    b,nkv,sn,hd=hs.shape
    if nr==1:return hs
    hs=hs[:,:,None,:,:].expand(b,nkv,nr,sn,hd)
    return hs.reshape(b,nkv*nr,sn,hd)
class MQA(nn.Module):
    def __init__(self,args:modelargs):
        """Initializes attention mechanism with QK normalization."""
        super().__init__()
        self.nh=args.n_heads
        self.nkv=args.n_kv_heads
        self.nr=self.nh//self.nkv
        self.hd=args.dim//args.n_heads
        self.wq=nn.Linear(args.dim,self.nh*self.hd,bias=False)
        self.wk=nn.Linear(args.dim,self.nkv*self.hd,bias=False)
        self.wv=nn.Linear(args.dim,self.nkv*self.hd,bias=False)
        self.wo=nn.Linear(self.nh*self.hd,args.dim,bias=False)
        self.qn=nn.RMSNorm(self.hd,eps=args.norm_eps)
        self.kn=nn.RMSNorm(self.hd,eps=args.norm_eps)
    def forward(self,ip:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor,msk:torch.Tensor|None=None,kvc:KVcache|None=None)->torch.Tensor:
        """Executes forward pass calculating attention scores."""
        bn,sn,_=ip.shape
        iq,ik,iv=self.wq(ip),self.wk(ip),self.wv(ip)
        iq=iq.view(bn,sn,self.nh,self.hd)
        ik=ik.view(bn,sn,self.nkv,self.hd)
        iv=iv.view(bn,sn,self.nkv,self.hd).transpose(1,2)
        iq=self.qn(iq).transpose(1,2)
        ik=self.kn(ik).transpose(1,2)
        iq,ik=rotary_embedding(iq,ik,cos,sin)
        if kvc is not None:ik,iv=kvc.change(ik,iv)
        ik=rpt_kv(ik,self.nr)
        iv=rpt_kv(iv,self.nr)
        sc=torch.matmul(iq,ik.transpose(2,3))/math.sqrt(self.hd)
        if msk is not None:sc=sc+msk
        sc=torch.softmax(sc.float(),dim=-1).type_as(iq)
        op=torch.matmul(sc,iv)
        op=op.transpose(1,2).contiguous().view(bn,sn,-1)
        return self.wo(op)