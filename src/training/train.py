import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from contextlib import nullcontext
from args import modelargs
from model import GIBCS
from dataset import mydataset
def st_ddp():
    """Initializes distributed data parallel environment"""
    dist.init_process_group(backend="nccl")
    lr=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    return lr
def train():
    """Training loop"""
    lr=st_ddp()
    d=torch.device(f"cuda:{lr}")
    a=modelargs()
    m=GIBCS(a).to(d)
    m=DDP(m,device_ids=[lr])
    td=os.path.dirname(os.path.abspath(__file__))
    tnd=os.path.join(td,"..","tokenizer")
    cd=os.path.join(tnd,"corpus")
    tp=os.path.join(tnd,"tokenizer.model")
    dp=os.path.join(cd,"tokenizer_corpus.txt")
    if lr==1:
        print("[Rank 1] Waiting for Rank 0",flush=True)
        dist.barrier()
    ds=mydataset(a,dp=dp,tp=tp)
    if lr==0:
        print("[Rank 0] Tokenization done",flush=True)
        dist.barrier()
    dist.barrier()
    smp=DistributedSampler(ds)
    bs=16
    acn=8
    dl=DataLoader(ds,batch_size=bs,sampler=smp,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(params=m.parameters(),lr=0.0005,weight_decay=0.01)
    scl=torch.amp.GradScaler()
    ep=3
    m.train()
    tb=len(dl)
    for e in range(ep):
        smp.set_epoch(e)
        opt.zero_grad()
        for bid,(i,j) in enumerate(dl):
            i,j=i.to(d),j.to(d)
            gac=(bid+1)%acn!=0 and (bid+1)!=tb
            gsy=m.no_sync() if gac else nullcontext()
            with gsy:
                with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
                    lts,ls=m(i,j)
                    ls=ls/acn
                scl.scale(ls).backward()
            if not gac:
                scl.step(opt)
                scl.update()
                opt.zero_grad()
            if lr==0 and bid%10==0:
                tls=ls.item()*acn
                print(f"EP:{e}/{ep}, BATCH:{bid}/{tb}, LOSS:{tls:.4f}")
    if lr==0:torch.save({'model_state':m.module.state_dict()},"gibcs_chckpt.pt")
    dist.destroy_process_group()
if __name__=="__main__":
    train()