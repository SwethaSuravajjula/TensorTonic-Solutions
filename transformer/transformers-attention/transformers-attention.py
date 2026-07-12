import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2,-1)
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = torch.softmax(scaled_scores,-1)
    attention_vector = attention_weights @ V
    
    return attention_vector
