import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    # YOUR CODE HERE
    x = np.array(x) # (1,2)
    conv1 = np.array(conv1) #(2,2)
    W1_b1 = np.array(W1_b1) #(2,2)
    W1_b2 = np.array(W1_b2) # (2,3)
    W2_b1 = np.array(W2_b1) #(2,2)
    W2_b2 = np.array(W2_b2) #(2,3)
    Ws_b2 = np.array(Ws_b2) #(2,3)
    fc = np.array(fc) #(3,2)

    h = x @ conv1  #(1,2)
    h = np.maximum(h,0) # (1,2)
    #block1
    
    b1_h1 = np.maximum(h @ W1_b1,0) # (1,2)
    b1_h2 = b1_h1 @ W2_b1 #(1,2)

    
    b1 = np.maximum(b1_h2 + h, 0) #(1,2)

    #block2 

    b2_h1 = np.maximum(b1 @ W1_b2,0)# (1,3)
    b2_h2 = b2_h1 @ W2_b2 #(1,3) 

    b2 = np.maximum(b2_h2 + b1 @ Ws_b2,0) # (1,3)

    # fully connected layer 
    y = b2 @ fc  #(1,2)

    return y
    
    
    

    
    

    
    
    
