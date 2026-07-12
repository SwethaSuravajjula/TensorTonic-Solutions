import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens =  [self.pad_token,self.unk_token,self.bos_token,self.eos_token]
        for special_token in special_tokens:
            if special_token not in self.word_to_id:
                self.word_to_id[special_token] = self.vocab_size
                self.id_to_word[self.vocab_size] = special_token
                self.vocab_size = self.vocab_size + 1
        unique_words = set()
        for text in texts:
            text = text.lower()
            text = text.split()
            unique_words.update(text)
        sorted_unique_words = sorted(unique_words)
        for word in sorted_unique_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size = self.vocab_size + 1
         
            
            
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        
        # YOUR CODE HERE
        text = text.lower().split()
        encoding = []

        for word in text:
            unk_token_id = self.word_to_id[self.unk_token]
            token = self.word_to_id.get(word, unk_token_id)
            encoding.append(token)
        return encoding 
            
            
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoding_tokens = []
        # YOUR CODE HERE
        for token_id in ids:
            word = self.id_to_word.get(token_id,self.unk_token)
            decoding_tokens.append(word)
        return " ".join(decoding_tokens)
        
