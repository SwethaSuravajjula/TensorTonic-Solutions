import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    eps = 1e-5
    
    
    
    # YOUR CODE HERE
    if mode == "post":
        conv1 = x @ W1
        var1 = np.var(conv1,axis=0)
        dem1 = np.sqrt(var1 + eps)
        bn1 = (conv1 - np.mean(conv1,axis=0)) / dem1
        h1 = np.maximum(gamma1 * bn1 + beta1,0)
        conv2 = h1 @ W2
        var2 = np.var(conv2,axis=0)
        dem2 = np.sqrt(var2 + eps)
        bn2 = (conv2-np.mean(conv2,axis=0)) / dem2
        h2 = gamma2 * bn2 + beta2
        y = h2 + x 
        return { 'output': np.maximum(y,0), 'mode': 'post' }
        
    if mode == "pre":
        var1 = np.var(x,axis=0)
        dem1 = np.sqrt(var1 + eps)
        bn1 = (x - np.mean(x,axis=0)) / dem1
        h1 = np.maximum(gamma1 * bn1 + beta1,0)
        conv1 = h1 @ W1
        var2 = np.var(conv1,axis=0)
        dem2 = np.sqrt(var2 + eps)
        bn2 = (conv1-np.mean(conv1,axis=0)) / dem2
        h2 = np.maximum(gamma2 * bn2 + beta2,0)
        y = h2 @ W2
        return { 'output': y + x , 'mode': 'pre' }

    
        
        
    
    
    
