import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code 
    # batch_size, seq_length , dimenstions
    dims = W_q.shape[-1] // num_heads # d_model / num_heads
    Q_projected = Q @ W_q   # (batch_size, seq_len, d_model) @ (d_model,d_model) -> (batch_size, seq_len, d_model)
    K_projected = K @ W_k   # (batch_size, seq_len, d_model) @ (d_model,d_model) -> (batch_size, seq_len, d_model)
    V_projected = V @ W_v   # (batch_size, seq_len, d_model) @ (d_model,d_model) -> (batch_size, seq_len, d_model)
    batch_size = Q_projected.shape[0]  
    seq_len = Q_projected.shape[1]
    
    Q_heads = Q_projected.reshape(batch_size, seq_len, num_heads, dims)  
    K_heads = K_projected.reshape(batch_size, seq_len, num_heads, dims)
    V_heads = V_projected.reshape(batch_size, seq_len, num_heads, dims)

    # (0,1,2,3) -> (batch_size, seq_len, num_head, dims)
    Q_heads =  Q_heads.transpose(0,2,1,3)  # (batch_size, num_head, seq_len, dims)
    K_heads =  K_heads.transpose(0,2,1,3)  # (batch_size, num_head, seq_len, dims)
    V_heads =  V_heads.transpose(0,2,1,3)  # (batch_size, num_head, seq_len, dims)
    # (0,1,2,3) -> (batch_size, num_heads, seq_len, dims)
    scores = ( Q_heads @ K_heads.transpose(0,1,3,2) ) / np.sqrt(dims)   # (batch_size,num_head,seq_len,dims) @ (batch_size, num_heads, dim, seq_len)  -> (batch_size, num_head, seq_len, seq_len)
    attention_scores = softmax(scores) @ V_heads # (batch_size, num_head, seq_len, seq_len) @ (batch_size, num_head, seq_len,dims) -> (batch_size, num_head,seq_len,dims)
    attention_scores = attention_scores.transpose(0,2,1,3) # (batch_size, seq_len, num_head,dims)
    combined = attention_scores.reshape(batch_size, seq_len, W_q.shape[-1]) # (batch_size, seq_len, d_model)
    
    output = combined @ W_o # (batch_size, seq_len, d_model) @ (d_model,d_model) -> (batch_size, seq_len, d_model)

    return output
    
    