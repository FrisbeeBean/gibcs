import torch
import sentencepiece as spm
from array import array
from torch.utils.data import Dataset
from args import modelargs
class mydataset(Dataset):
    def __init__(self,args:modelargs,dp:str,tp:str):
        super().__init__()
        self.sn=args.max_seq_n
        self.bid=1
        self.eid=2
        self.sp=spm.SentencePieceProcessor(model_file=tp)
        at=array('h')
        with open(dp,"r",encoding="utf-8") as f:
            b=[]
            for l in f:
                r=l.strip()
                if len(r)>1:b.append(r)
                if len(b)>100000:
                    eb=self.sp.Encode(b)
                    for t in eb:
                        at.append(self.bid)
                        at.extend(t)
                        at.append(self.eid)
                    b=[]
            if b:
                eb=self.sp.Encode(b)
                for t in eb:
                    at.append(self.bid)
                    at.extend(t)
                    at.append(self.eid)
        self.d=torch.tensor(at,dtype=torch.long)
        self.ops=len(self.d)//(self.sn+1)
    def __len__(self):
        return self.ops
    def __getitem__(self,id):
        bi=id*(self.sn+1)
        ei=bi+self.sn+1
        c=self.d[bi:ei]
        return c[:-1],c[1:]