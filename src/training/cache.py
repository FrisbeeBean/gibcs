import torch
from args import modelargs

class KVcache:
    def __init__(self,args:modelargs,device:torch.device):
        self.head_dim=args.dim//args.n_heads
        self.k_cache=torch.zeros((args.max_batch_n,args.n_kv_heads,args.max_seq_n,self.head_dim),dtype=torch.float16,device=device)
        self.v_cache=torch.zeros((args.max_batch_n,args.n_kv_heads,args.max_seq_n,self.head_dim),dtype=torch.float16,device=device)
        # we are using type float16 because i am training on gpu t4x2 and T4 is optimized for 
        # FP16 precision 
        self.seq_n=0

    def change(self,k:torch.Tensor,v:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """This is used to update the KV cache and it returns a tuple of all cached key and value"""
        batch_n,_,seq_n,_=k.shape
        k=k.to(dtype=torch.float16)
        v=v.to(dtype=torch.float16)
        self.k_cache[:batch_n,:,self.seq_n:self.seq_n+seq_n,:]=k
        self.v_cache[:batch_n,:,self.seq_n:self.seq_n+seq_n,:]=v
        self.seq_n+=seq_n
        return (self.k_cache[:batch_n,:,:self.seq_n,:],self.v_cache[:batch_n,:,:self.seq_n,:])