import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.api.model import LM
from inference import loader
from tqdm import tqdm
from datasets import load_dataset

class GIBCSEval(LM):
    def __init__(self, cp="gibcs_chckpt.pt", tp="../tokenizer/tokenizer.model"):
        """Wraps model for lm-evaluation-harness compatibility"""
        super().__init__()
        self.b = loader(cp, tp)
        self.m = self.b.m
        self.sp = self.b.sp
        self.d = self.b.d
    @property
    def eot_token_id(self): return 2
    @property
    def max_length(self): return 1024
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return 1
    @property
    def device(self): return self.d

    def tok_encode(self, s): return self.sp.Encode(s)
    def tok_decode(self, t): return self.sp.Decode(t)
    def loglikelihood(self, rqs):
        """Computes log-probabilities for multiple-choice tasks (ARC, HellaSwag, etc)"""
        rs = []
        for rq in tqdm(rqs, desc="Evaluating MCQs"):
            cx, cn = rq.args
            cxt = self.sp.Encode(cx)
            cnt = self.sp.Encode(cn)
            if not cxt: cxt = [1]
            ft = cxt + cnt
            ip = torch.tensor(ft[:-1], dtype=torch.long, device=self.d).unsqueeze(0)
            with torch.no_grad(): lts, _ = self.m(ip)
            clts = lts[0, len(cxt) - 1:, :]
            ctgt = torch.tensor(cnt, dtype=torch.long, device=self.d)
            lp = F.log_softmax(clts, dim=-1)
            glp = torch.gather(lp, 1, ctgt.unsqueeze(1)).squeeze(1)
            ig = (lp.argmax(dim=-1) == ctgt).all().item()
            rs.append((glp.sum().item(), ig))
        return rs
    def generate_until(self, rqs):
        return [""] * len(rqs)

    @torch.no_grad()
    def wikitext_perplexity(self, max_length=None, stride=512, limit_tokens=None):
        max_length = max_length or self.max_length
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if t.strip() != ""])
        if limit_tokens is not None:
            text = text[: limit_tokens * 6]
        word_count = len(text.split())
        print(f"Held-out text: {word_count} words")
        enc = self.sp.Encode(text)
        input_ids = torch.tensor(enc, dtype=torch.long, device=self.d)
        seq_len = input_ids.size(0)
        print(f"Encoded length: {seq_len} tokens "
              f"(fertility ~{seq_len / max(word_count,1):.2f} tokens/word)")
        nll_sum = 0.0
        n_scored_tokens = 0
        prev_end = 0
        pbar = tqdm(range(0, seq_len, stride), desc="Computing perplexity")
        for begin in pbar:
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end  
            if trg_len <= 0:
                if end == seq_len:
                    break
                continue
            window_ids = input_ids[begin:end]
            if window_ids.size(0) < 2:
                break
            ip = window_ids[:-1].unsqueeze(0)
            target = window_ids[1:]
            logits, _ = self.m(ip)
            logits = logits[0]  
            n_new = min(trg_len, target.size(0))
            logits_new = logits[-n_new:]
            target_new = target[-n_new:]
            log_probs = F.log_softmax(logits_new, dim=-1)
            token_nll = -torch.gather(log_probs, 1, target_new.unsqueeze(1)).squeeze(1)
            nll_sum += token_nll.sum().item()
            n_scored_tokens += n_new
            prev_end = end
            pbar.set_postfix(ppl_so_far=f"{torch.exp(torch.tensor(nll_sum / max(n_scored_tokens,1))).item():.3f}")
            if end == seq_len:
                break

        avg_nll_per_word = nll_sum / word_count
        ppl = torch.exp(torch.tensor(avg_nll_per_word)).item()

        token_level_ppl = torch.exp(torch.tensor(nll_sum / n_scored_tokens)).item()

        print(f"\nScored {n_scored_tokens} tokens over {word_count} words")
        print(f"Token-level PPL (NOT comparable across tokenizers): {token_level_ppl:.4f}")
        print(f"Word-level PPL   (comparable, use THIS number):      {ppl:.4f}")
        return {"word_level_ppl": ppl, "token_level_ppl": token_level_ppl,
                "n_tokens": n_scored_tokens, "n_words": word_count}

def main():
    """Runs evaluation benchmark suite"""
    print("Loading Model for Evaluation...")
    m = GIBCSEval()
    tsks = ["arc_easy", "hellaswag", "piqa", "winogrande"]
    print(f"Starting lm-evaluation-harness on {tsks}...")
    rs = lm_eval.simple_evaluate(model=m, tasks=tsks, batch_size=1)
    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS:")
    print("=" * 50)
    for t, r in rs["results"].items():
        print(f"{t.upper()} (Accuracy): {r.get('acc,none', r.get('acc', 0)) * 100:.2f}%")
    print("\nRunning WikiText-103 perplexity ")
    ppl_results = m.wikitext_perplexity(max_length=1024, stride=512)
    print(f"\nWIKITEXT-103 (Word-level Perplexity): {ppl_results['word_level_ppl']:.4f}")

if __name__ == "__main__":
    main()