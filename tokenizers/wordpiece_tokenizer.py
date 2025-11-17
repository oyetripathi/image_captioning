import re
import json
from tokenizers.base_tokenizer import BaseTokenizer

def preprocess_sentence(txt):
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9., ]", "", txt)
    txt = re.sub(r"([.,])", r" \1 ", txt)
    txt = re.sub(r"([a-z])([0-9])", r"\1 \2", txt)
    txt = re.sub(r"([0-9])([a-z])", r"\1 \2", txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.strip()    
    return txt


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, max_len=250):
        super().__init__(max_len)
        self.vocab = None
        self.id2word = None
        self.pad_token_id = None
        self.vocab_size = 0
    
    def build_vocab(self, vocab_path=None):
        if vocab_path is not None:
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)
        else:
            raise Exception("Please provide a valid vocab_path to load vocabulary, wordpiece tokenizer must be pre-trained")
        
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab["<pad>"]
        self.id2word = {i: w for w,i in self.vocab.items()}
    
    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i>0 and not (word[:i] in self.vocab): i-=1
            if i==0: return [self.vocab["<unk>"]]
            tokens.append(self.vocab[word[:i]])
            word = word[i:]
            if len(word):
                word = f"##{word}"
        return tokens
    
    def encode(self, text):
        if self.vocab is None:
            raise Exception("Please build vocabulary first")
        
        words = ["<start>"]+ preprocess_sentence(text).split() + ["<end>"]
        
        token_ids = []
        for word in words:
            token_ids.extend(self.encode_word(word))
        token_ids = token_ids[:self.max_len]
        mask = [(token_id != self.vocab.get("<pad>")) for token_id in token_ids]
        return token_ids, mask
    
    def decode(self, token_ids):
        decoded_tokens = [self.id2word.get(idx, "<unk>") for idx in token_ids]
        decoded_sentence = ""
        i = 0
        running_word = ""
        for word in decoded_tokens:
            if word in ["<start>", "<end>", "<pad>", "<unk>"]:
                continue
            if word.startswith("##"): running_word += word.removeprefix("##")
            else:
                decoded_sentence += (running_word + " ")
                running_word = word
        decoded_sentence += (running_word + " ")
        return " ".join([x.strip() for x in decoded_sentence.strip().split()])
    
    def save(self, file_path):
        if self.vocab is None:
            raise Exception("Please build vocabulary first")
        
        with open(file_path, "w") as f:
            json.dump(self.vocab, f)
    
    def load(self, file_path):
        with open(file_path, "r") as f:
            self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)
            self.pad_token_id = self.vocab["<pad>"]
            self.id2word = {i: w for w,i in self.vocab.items()}