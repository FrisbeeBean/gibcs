import torch
import torch.nn as nn
import torch.nn.functional as F
from args import modelargs
from rope import RoPE
from attention import MQA
from dense import SwiGLU
from cache import KVcache
class trnsfrmr(nn.Module):
    def __init__(self,args:modelargs):
        """Initializes single transformer block"""
        super().__init__()
        self.an=nn.RMSNorm(args.dim,eps=args.norm_eps)
        self.attn=MQA(args)
        self.fn=nn.RMSNorm(args.dim,eps=args.norm_eps)
        self.lyr=SwiGLU(args)
    def forward(self,ip:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor,msk:torch.Tensor|None=None,kvc:KVcache|None=None)->torch.Tensor:
        """Executes transformer block"""
        at=self.attn(self.an(ip),cos,sin,msk,kvc)
        rs=ip+at
        op=rs+self.lyr(self.fn(rs))
        return op
class GIBCS(nn.Module):
    def __init__(self,args:modelargs):
        """Initializes LLM architecture"""
        super().__init__()
        self.a=args
        self.te=nn.Embedding(args.vocab_n,args.dim)
        self.rp=RoPE(args.dim//args.n_heads,args)
        self.ls=nn.ModuleList([trnsfrmr(args) for _ in range(args.n_layers)])
        self.rn=nn.RMSNorm(args.dim,eps=args.norm_eps)
        self.op=nn.Linear(args.dim,args.vocab_n,bias=False)
        self.op.weight=self.te.weight
        self.apply(self._init_w)
    def _init_w(self,m):
        """Initializes weights"""
        if isinstance(m,nn.Linear):
            torch.nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None:torch.nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            torch.nn.init.normal_(m.weight,mean=0.0,std=0.02)
    def forward(self,tkns:torch.Tensor,tgts:torch.Tensor|None=None,kvcs:list[KVcache]|None=None)->tuple[torch.Tensor,torch.Tensor|None]:
        """Executes full pass computing logits and loss"""
        _,sn=tkns.shape
        t=self.te(tkns)
        sp=kvcs[0].seq_n if kvcs is not None else 0
        cos,sin=self.rp(t,sn+sp)
        cos=cos[:,:,sp:sp+sn,:]
        sin=sin[:,:,sp:sp+sn,:]
        msk=None
        if sn>1:msk=torch.triu(torch.full((sn,sn),float("-inf"),device=tkns.device),diagonal=1)
        for i,l in enumerate(self.ls):
            c=kvcs[i] if kvcs is not None else None
            t=l(t,cos,sin,msk,c)
        lts=self.op(self.rn(t))
        ls=None
        if tgts is not None:ls=F.cross_entropy(lts.view(-1,self.a.vocab_n),tgts.view(-1))
        return lts,ls