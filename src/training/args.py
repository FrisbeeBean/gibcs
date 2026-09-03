from dataclasses import dataclass
@dataclass
class modelargs:
    dim:int=512
    n_layers:int=14
    n_heads:int=8
    n_kv_heads:int=1
    vocab_n:int=16384
    hidden_dim:int=1536
    max_seq_n:int=1024
    max_batch_n:int=32
    norm_eps:float=1e-5