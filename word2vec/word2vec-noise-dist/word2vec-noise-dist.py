import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    # YOUR CODE HERE
    counts = counts.to(torch.float32)
    probs = counts ** alpha / (counts ** alpha).sum()
    return probs
    
