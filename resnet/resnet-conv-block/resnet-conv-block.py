import numpy as np

# def conv_block(x, W1, W2, Ws):
#     """
#     Returns: np.ndarray with sum of main path output and projected shortcut
#     """
#     h = np.maximum(np.array(x) @ np.array(W1),0)
#     z = np.maximum(h @ np.array(W2),0)
#     s = np.array(x) @ np.array(Ws)
#     y = s + z

#     return y
def conv_block(x, W1, W2, Ws):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)

    h = np.maximum(x @ W1, 0)
    z = h @ W2

    s = x @ Ws

    y = np.maximum(z + s,0)

    return y
    
    
