import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    # x dimension -> (batch_size, seq_len, d_model), W1 -> (d_model,d_model) -> (batch_size, seq_length, d_model)
    y_hidden = x @ W1 + b1
    # 
    y_hidden = np.maximum(0, y_hidden) @ W2 + b2

    return y_hidden
    
    