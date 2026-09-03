import torch
import torch.nn.functional as F
import sentencepiece as spm
from args import modelargs
from model import GIBCS
class loader:
    def __init__(self,cp,tp,d="cuda" if torch.cuda.is_available() else "cpu"):
        """Loads model weights and tokenizer for inference"""
        self.d=d
        self.a=modelargs()
        self.sp=spm.SentencePieceProcessor(tp)
        self.bid=1
        self.eid=2
        self.m=GIBCS(self.a).to(d)
        c=torch.load(cp,map_location=self.d,weights_only=True)
        self.m.load_state_dict(c['model_state'])
        self.m.eval()
    @torch.no_grad()
    def gen(self,prompt,max_tokens=200,temperature=0.5,top_k=40,top_p=0.9):
        """Generates text from given prompt"""
        tks=self.sp.Encode(prompt)
        tks=torch.tensor(tks,dtype=torch.long,device=self.d).unsqueeze(0)
        gtk=[]
        for _ in range(max_tokens):
            ctx=tks[:,-self.a.max_seq_n:]
            lts,_=self.m(ctx,tgts=None)
            pl=lts[:,-1,:]
            if temperature>0.0:pl=pl/temperature
            if top_k>0:
                tv,_=torch.topk(pl,k=top_k,dim=-1)
                pl[pl<tv[:,[-1]]]=-float('Inf')
            prb=F.softmax(pl,dim=-1)
            if top_p>0.0:
                spb,si=torch.sort(prb,dim=-1,descending=True)
                cpb=torch.cumsum(spb,dim=-1)
                rmv=cpb>top_p
                rmv[...,1:]=rmv[...,:-1].clone()
                rmv[...,0]=0
                trmv=rmv.scatter(1,si,rmv)
                prb[trmv]=0.0
                prb=prb/prb.sum(dim=-1,keepdim=True)
            ntk=torch.multinomial(prb,num_samples=1)
            if ntk.item()==self.eid:break
            gtk.append(ntk.item())
            tks=torch.cat((tks,ntk),dim=1)
        return self.sp.Decode(gtk)