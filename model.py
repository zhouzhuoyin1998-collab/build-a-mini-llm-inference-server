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

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.
    logits = np.asarray(logits, dtype=float)

    # Step 1: Calculate probabilities
    probs = stable_softmax(logits)

    # Step 2: Sort descending, preserving ties
    indices = np.argsort(-probs, axis=-1, kind="stable")

    # Step 3: Get sorted probabilities
    sorted_probs = np.take_along_axis(
        probs, indices, axis=-1
    )

    # Step 4: Cumulative probability
    cumulative = np.cumsum(sorted_probs, axis=-1)

    # Step 5: Keep tokens until cumulative probability >= p
    keep = (cumulative - sorted_probs) < p

    # Step 6: Sort original logits
    sorted_logits = np.take_along_axis(
        logits, indices, axis=-1
    )

    # Step 7: Filter
    filtered = np.where(keep, sorted_logits, -np.inf)

    # Step 8: Restore original order
    original_order = np.argsort(indices, axis=-1)

    return np.take_along_axis(
        filtered, original_order, axis=-1
    )

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    return int(rng.choice(len(probs), p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    return int(np.argmax(logits))

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    token_to_id = {}
    id_to_token = []

    # 1. Add special tokens first
    for token in special_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)

    # 2. Get all unique characters from corpus
    chars = set()

    for text in corpus:
        for char in text:
            chars.add(char)

    # 3. Add remaining characters in sorted order
    for char in sorted(chars):
        if char not in token_to_id:
            token_to_id[char] = len(id_to_token)
            id_to_token.append(char)

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token
    }

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    token_to_id = vocab["token_to_id"]
    result = []

    if add_bos and "<bos>" in token_to_id:
        result.append(token_to_id["<bos>"])

    for char in text:
        if char in token_to_id:
            result.append(token_to_id[char])

        elif "<unk>" in token_to_id:
            result.append(token_to_id["<unk>"])

    return result

