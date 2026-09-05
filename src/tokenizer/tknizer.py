import sentencepiece as spm
def main():
    """Train BPE tokenizer"""
    dp="corpus/tokenizer_corpus.txt"
    vs=16384
    spm.SentencePieceTrainer.Train(input=dp,model_prefix="tokenizer",
                                   vocab_size=vs,model_type="bpe",
                                   byte_fallback=True,unk_piece="<unk>",
                                   pad_piece="<pad>",bos_piece="<s>",
                                   eos_piece="</s>",
                                   train_extremely_large_corpus=True,
                                   normalization_rule_name="identity",
                                   max_sentence_length=16384)
if __name__=="__main__":
    main()