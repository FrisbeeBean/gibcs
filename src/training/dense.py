import torch.nn as nn
from args import modelargs
class SwiGLU(nn.Module):
    def __init__(self,args:modelargs):
        super().__init__()
        self.w1=nn.Linear(args.dim,args.hidden_dim,bias=False)
        self.w2=nn.Linear(args.hidden_dim,args.dim,bias=False)
        self.w3=nn.Linear(args.dim,args.hidden_dim,bias=False)
        self.silu=nn.SiLU()
    def forward(self,ip):
        return self.w2(self.silu(self.w1(ip)*self.w3(ip)))