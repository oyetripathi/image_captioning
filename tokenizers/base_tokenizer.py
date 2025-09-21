class BaseTokenizer:
    def __init__(self, max_len=30):
        self.max_len = max_len
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, text):
        raise NotImplementedError
    
    def save(self, file_path):
        raise NotImplementedError
    
    def load(self, file_path):
        raise NotImplementedError