import numpy as np


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Apply layer normalization across the last dimension.

    x:     (batch_size, seq_len, d_model)
    gamma: (d_model,)
    beta:  (d_model,)

    Returns:
        Array with the same shape as x.
    """

    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalized = (x - mean) / np.sqrt(variance + eps)

    output = gamma * normalized + beta

    return output