import os
from datasets import load_dataset
def strm_sv(dp,nm,spt,of,tc="text",mr=2000000):
    """Stream dataset and save text to file"""
    ds=load_dataset(dp,name=nm,split=spt,streaming=True)
    c=0
    with open(of,"w",encoding="utf-8") as f:
        for r in ds:
            if c>=mr:break
            t=r[tc].strip()
            if t:
                f.write(t+"\n\n")
                c+=1
def main():
    """Download datasets and create tokenizer corpus"""
    os.makedirs("corpus",exist_ok=True)
    strm_sv("roneneldan/TinyStories",None,"train","corpus/tinystories.txt",mr=2000000)
    strm_sv("HuggingFaceFW/fineweb-edu","sample-10BT","train","corpus/fineweb_edu.txt",mr=2000000)
    with open("corpus/tokenizer_corpus.txt","w",encoding="utf-8") as outf:
        with open("corpus/tinystories.txt","r",encoding="utf-8") as f1:
            for i,l in enumerate(f1):
                if i<1000000:outf.write(l)
                else:break
        with open("corpus/fineweb_edu.txt","r",encoding="utf-8") as f2:
            for i,l in enumerate(f2):
                if i<1000000:outf.write(l)
                else:break
if __name__=="__main__":
    main()