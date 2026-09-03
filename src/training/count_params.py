from args import modelargs
from model import GIBCS
def cnt_p(m):
    up=set(m.parameters())
    tp=sum(p.numel() for p in up if p.requires_grad)
    return tp
def main():
    """Initializes model and verifies total parameters are under 50M."""
    a=modelargs()
    m=GIBCS(a)
    tp=cnt_p(m)
    print(f"Config:{a}")
    print(f"Total Trainable Parameters:{tp:,}")
    assert tp<=50000000,"Model exceeds 50M parameter limit!"
    print("SUCCESS:Model is under the 50,000,000 parameter limit.")
if __name__=="__main__":
    main()