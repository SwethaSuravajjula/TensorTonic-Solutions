def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    
    f0 = 2*a*x + b
    x = x0 - lr*f0 
    f = 2*a*x + b
    for i in range(1,steps):
        x = x - lr*f
        f = 2*a*x + b
    return x