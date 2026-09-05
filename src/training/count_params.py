from args import modelargs
from model import GIBCS
def cnt_p(m):
    up=set(m.parameters())
    tp=sum(p.numel() for p in up if p.requires_grad)
    return tp
def main():
    """Initializes model and prints total parameter count"""
    a=modelargs()
    m=GIBCS(a)
    tp=cnt_p(m)
    print(f"Config:{a}")
    print(f"Total Trainable Parameters:{tp:,}")
if __name__=="__main__":
    main()