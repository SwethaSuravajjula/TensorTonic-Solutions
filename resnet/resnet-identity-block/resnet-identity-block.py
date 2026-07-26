import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    W1 = np.array(W1)
    W2 = np.array(W2)
    x_h = np.maximum(0,x @ W1.T)
    y = np.maximum(((x_h @ W2.T) + x),0)
    
    return y
    
