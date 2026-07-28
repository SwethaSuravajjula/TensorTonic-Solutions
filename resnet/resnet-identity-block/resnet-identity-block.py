import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    
    h = np.maximum(0,np.array(x) @ np.array(W1).T)
    y = np.maximum(0, (h @ np.array(W2).T) + np.array(x))
    return y
    
