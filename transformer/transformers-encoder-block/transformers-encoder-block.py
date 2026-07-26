import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x,axis = -1, keepdims=True)
    sigma = np.var(x,axis= -1,keepdims=True)
    normalized = (x - mu) / np.sqrt(sigma + eps)
    return gamma * normalized + beta
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    dims = W_q.shape[-1] // num_heads
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]
    Q_projected = Q @ W_q
    K_projected = K @ W_k
    V_projected = V @ W_v

    Q_heads = Q_projected.reshape(batch_size,seq_len,num_heads,dims)
    K_heads = K_projected.reshape(batch_size,seq_len,num_heads,dims)
    V_heads = V_projected.reshape(batch_size,seq_len,num_heads,dims)

    Q_heads = Q_heads.transpose(0,2,1,3)
    K_heads = K_heads.transpose(0,2,1,3)
    V_heads = V_heads.transpose(0,2,1,3)

    scores = (Q_heads @ K_heads.transpose(0,1,3,2)) / np.sqrt(dims)
    attention_scores = softmax(scores) @ V_heads

    attention_scores = attention_scores.transpose(0,2,1,3)

    attention_scores = attention_scores.reshape(batch_size,seq_len,W_q.shape[-1])

    return attention_scores @ W_o
    

    
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    x = x @ W1 + b1
    x = np.maximum(0,x) @ W2 + b2

    return x
    
    # Your code here
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    

    x1 = multi_head_attention(x,x,x,W_q,W_k,W_v, W_o, num_heads)
    x2 = layer_norm(x+x1,gamma1,beta1)
    x3 = feed_forward(x2,W1,b1,W2,b2)
    y = layer_norm(x2+x3,gamma2,beta2)

    return y
    
    
    