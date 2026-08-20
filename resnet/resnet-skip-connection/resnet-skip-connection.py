import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    if gradients_F == []:
        return np.array(x)
    gradients_f = np.array(gradients_F)
    result = np.eye(gradients_f.shape[1])
    for jacobians in range(0,gradients_f.shape[0]):
        result = result @ (gradients_f[jacobians] + np.eye(gradients_f.shape[1]))
    result = x @ result
    
    return result
        
        
    # YOUR CODE HERE
    

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    if gradients_F == []:
        return np.array(x)
    gradients_f = np.array(gradients_F)
    result = np.eye(gradients_f.shape[1])
    for jacobians in range(0,gradients_f.shape[0]):
        result = result @ gradients_f[jacobians]
    result = x @ result
    
    return result
    
