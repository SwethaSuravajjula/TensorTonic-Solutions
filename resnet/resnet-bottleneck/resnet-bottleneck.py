import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)
    x = np.array(x)

    h1 = np.maximum(x @ W1,0)
    h2 = np.maximum(h1 @ W2, 0)
    h3 = h2 @ W3

    if h3.shape[1] != x.shape[1]:
        return np.maximum(( x @ Ws ) + h3, 0)
    else:
        return np.maximum(h3 + x,0)
    
