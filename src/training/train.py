import os
import time
import json
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from contextlib import nullcontext
from args import modelargs
from model import GIBCS
from dataset import mydataset
# Used 2 gpu T4, therefore i used data parallelism so DistributedDataParallel was required
def st_ddp():
    """Initializes distributed data parallel environment"""
    os.environ["OMP_NUM_THREADS"]="1"
    dist.init_process_group(backend="nccl")
    lr=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    return lr
def train():
    """Executes the training loop with checkpointing and loss tracking."""
    start_time=time.time()
    # Training on kaggle's GPU so keeping time limit of 9.5 hrs since session lasts only 12 hrs
    time_limit=9.5*3600 
    lr=st_ddp()
    d=torch.device(f"cuda:{lr}")
    a=modelargs()
    m=GIBCS(a).to(d)
    # Compiling model for faster training
    m=torch.compile(m)
    m=DDP(m,device_ids=[lr])
    # Navigating through repo structure 
    td=os.path.dirname(os.path.abspath(__file__))
    tnd=os.path.join(td,"..","tokenizer")
    cd=os.path.join(tnd,"corpus")
    tp=os.path.join(tnd,"tokenizer.model")
    dp=os.path.join(cd,"tokenizer_corpus.txt")
    cp=os.path.join(td,"gibcs_chckpt.pt")
    lhp=os.path.join(td,"loss_log.json")
    lp=os.path.join(td,"loss_curve.png")
    #Implemented checkpointing thought i would use it but never needed since trained only for one session
    lsh=[]
    if lr==0 and os.path.exists(lhp):
        with open(lhp,"r") as file:
            lsh=json.load(file)
    if os.path.exists(cp):
        if lr==0:
            print(f"Resuming from {cp}",flush=True)
        chk=torch.load(cp,map_location=d,weights_only=True)
        m.module.load_state_dict(chk["model_state"])
    ds=mydataset(a,dp=dp,tp=tp)
    bs=16
    acn=8
    dl=DataLoader(ds,batch_size=bs,num_workers=0,pin_memory=True)
    opt=torch.optim.AdamW(params=m.parameters(),lr=0.0005,weight_decay=0.01,fused=True)
    scl=torch.amp.GradScaler()
    m.train()
    for batch_id,(inputs,targets) in enumerate(dl):
        inputs,targets=inputs.to(d),targets.to(d)
        accumulate=(batch_id+1)%acn!=0
        sync_context=m.no_sync() if accumulate else nullcontext()
        with sync_context:
            with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
                _,loss=m(inputs,targets)
                loss=loss/acn
            scl.scale(loss).backward()
        if not accumulate:
            scl.step(opt)
            scl.update()
            opt.zero_grad()
        if lr==0 and batch_id%10==0:
            total_loss=loss.item()*acn
            print(f"BATCH:{batch_id}, LOSS:{total_loss:.4f}",flush=True)
            lsh.append(total_loss)
        if batch_id%50==0:
            stop=torch.tensor(0,device=d)
            if lr==0 and time.time()-start_time>time_limit:
                stop+=1
            dist.broadcast(stop,src=0)
            if stop.item()==1:
                if lr==0:
                    print("Time limit reached. Saving weights and plot...",flush=True)
                    torch.save({"model_state":m.module.state_dict()},cp)
                    with open(lhp,"w") as file:
                        json.dump(lsh,file)
                    plt.figure(figsize=(10,5))
                    plt.plot(lsh,label="Train Loss",color="blue",linewidth=1.5)
                    plt.title("GIBCS Training Curve")
                    plt.xlabel("Logging Steps (x10 Batches)")
                    plt.ylabel("Loss")
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(lp)
                    plt.close()
                dist.destroy_process_group()
                return
if __name__=="__main__":
    train()