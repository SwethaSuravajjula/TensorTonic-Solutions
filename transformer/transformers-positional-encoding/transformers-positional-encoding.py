import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    positional_matrix = np.zeros((seq_length,d_model))
    for token in range(seq_length):
        for dim in range(d_model):
            if dim % 2 == 0:
                pair = dim // 2
                positional_matrix[token][dim] = np.sin (token / 10000**(2*pair/d_model))
            else:
                positional_matrix[token][dim] = np.cos (token / 10000**(2*pair/d_model))
    return positional_matrix
            
    pass