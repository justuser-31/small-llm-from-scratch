import tiktoken
import torch

class SimpleTokenizer:
    def __init__(self, tokenizer_name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_ids_list):
        return [self.decode(token_ids) for token_ids in token_ids_list]
