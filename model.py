"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    logits = np.asarray(logits, dtype=float)

    # Step 1: Subtract the maximum for numerical stability
    shifted = logits - np.max(logits, axis=-1, keepdims=True)

    # Step 2: Exponentiate
    exp_z = np.exp(shifted)

    # Step 3: Normalize along the last axis
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    logits = np.array(logits)
    return logits*(1/temperature) if temperature > 0 else logits

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    # TODO: keep only the k largest logits along the last axis, set the rest to -inf
    logits = np.asarray(logits)

    vocab_size = logits.shape[-1]

    if k >= vocab_size:
        return logits.copy()

    # kth largest value in each row
    threshold = np.partition(
        logits,
        -k,
        axis=-1
    )[..., -k]

    # Add last dimension so it broadcasts for both 1D and 2D
    threshold = np.expand_dims(threshold, axis=-1)

    # Keep everything tied with or above kth largest
    mask = logits >= threshold

    result = np.full_like(logits, -np.inf)
    result[mask] = logits[mask]

    return result

