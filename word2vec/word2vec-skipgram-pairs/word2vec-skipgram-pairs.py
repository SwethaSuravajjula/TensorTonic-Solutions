import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    pairs = []
    n = token_ids.numel()
    for i in range(n): #center vector 
        left = max(0, i-window)
        right = min(n-1 , i+window)
        for j in range(left,right+1):
            if i == j:
                continue
            else:
                pairs.append([token_ids[i].item(), token_ids[j].item()])
                
    
    if len(pairs) == 0:
        return torch.empty(
            (0, 2),
            dtype=torch.int64,
            device=token_ids.device
        )

    return torch.tensor(
        pairs,
        dtype=torch.int64,
        device=token_ids.device
    )
