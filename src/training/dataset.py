import torch
import sentencepiece as spm
import os
from torch.utils.data import IterableDataset
from args import modelargs

class mydataset(IterableDataset):
    def __init__(self,args:modelargs,dp:str,tp:str):
        super().__init__()
        self.seq_n=args.max_seq_n
        self.bid=1
        self.eid=2
        self.dp=dp
        self.sp=spm.SentencePieceProcessor(model_file=tp)
        self.lr=int(os.environ.get("LOCAL_RANK","0"))
        self.ws=int(os.environ.get("WORLD_SIZE","1"))

    def __iter__(self):
        batch=[]
        tokens=[]
        with open(self.dp,"r",encoding="utf-8") as file:
            for line_id,line in enumerate(file):
                if line_id%self.ws!=self.lr:
                    continue
                text=line.strip()
                if len(text)>1:
                    batch.append(text)
                if len(batch)>=2000:
                    encoded_batch=self.sp.Encode(batch)
                    for encoded in encoded_batch:
                        tokens.append(self.bid)
                        tokens.extend(encoded)
                        tokens.append(self.eid)
                    batch=[]
                    while len(tokens)>=self.seq_n+1:
                        chunk=tokens[:self.seq_n+1]
                        tokens=tokens[self.seq_n:]
                        yield torch.tensor(chunk[:-1],dtype=torch.long),torch.tensor(chunk[1:],dtype=torch.long)