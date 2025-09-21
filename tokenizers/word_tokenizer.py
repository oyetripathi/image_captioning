import json
from tokenizers.base_tokenizer import BaseTokenizer
from tokenizers.build_vocab import build_word_vocab

class WordTokenizer(BaseTokenizer):
    def __init__(self, max_len=30):
        super().__init__(max_len)
        self.vocab = None
        self.id2word = None
    
    def build_vocab(self, df, **kwargs):
        self.vocab = build_word_vocab(df, **kwargs)
        self.id2word = {i: w for w,i in self.vocab.items()}
    
    def encode(self, text):

        if self.vocab is None:
            raise Exception("Please build vocabulary first")

        tokens = ["<start>"]+ text.lower().split() + ["<end>"]

        if len(tokens) < self.max_len:
            pad_len = self.max_len - len(tokens)
            tokens += ["<pad>"]*pad_len

        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens][:self.max_len]

        mask = [(token_id != self.vocab.get("<pad>")) for token_id in token_ids]

        return token_ids, mask
    
    def decode(self, token_ids):
        words = [self.id2word.get(i, "<unk>") for i in token_ids]
        return " ".join(words).replace("<end>", "").replace("<start>", "").replace("<pad>", "").strip()
    
    def save(self, file_path):
        if self.vocab is None:
            raise Exception("Please build vocabulary first")
        
        with open(file_path, "w") as f:
            json.dump(self.vocab, f)
    
    def load(self, file_path):
        with open(file_path, "r") as f:
            self.vocab = json.load(f)
            self.id2word = {i: w for w,i in self.vocab.items()}