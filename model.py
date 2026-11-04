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

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    id_to_token = vocab["id_to_token"]
    result = ""

    for token_id in token_ids:
        token = id_to_token[token_id]

        if skip_special and token.startswith("<") and token.endswith(">"):
            continue

        result += token

    return result

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return embedding_matrix[token_ids]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    result = x @ weight

    if bias is not None:
        result = result + bias

    return result

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    return {
        "K": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "V": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "length": 0
    }

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    start = cache["length"]

    # Step 2: Number of new tokens
    t = k_new.shape[0]

    # Step 3: Calculate ending position
    end = start + t

    # Step 4: Write new K and V into cache
    cache["K"][start:end] = k_new
    cache["V"][start:end] = v_new

    # Step 5: Update the length
    cache["length"] = end

    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    # TODO: scaled dot-product attention with optional causal mask, returns (Tq, D)
    d_k = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d_k)

    # Step 2: Apply causal mask
    if is_causal:
        Tq = q.shape[0]
        Tk = k.shape[0]

        i = np.arange(Tq)[:, None]
        j = np.arange(Tk)[None, :]

        mask = j > i + (Tk - Tq)

        scores = np.where(mask, -np.inf, scores)

    # Step 3: Softmax
    weights = stable_softmax(scores)

    # Step 4: Weighted sum of Values
    return weights @ v

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # TODO: embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits.
    # Step 1: Embed tokens
    x = embed_tokens(token_ids, params["embedding"])

    # Step 2: Q, K, V projections
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    # Step 3: Initialize KV cache
    cache = init_kv_cache(
        params["max_seq_len"],
        x.shape[-1]
    )

    # Step 4: Store K and V
    append_kv(cache, k, v)

    # Step 5: Causal attention
    attention = causal_attention(
        q,
        cache["K"][:cache["length"]],
        cache["V"][:cache["length"]],
        is_causal=True
    )

    # Step 6: Attention output projection
    hidden = linear_projection(attention, params["Wo"])

    # Step 7: Get last token representation
    last_hidden = hidden[-1]

    # Step 8: Project to vocabulary logits
    logits = linear_projection(
        last_hidden,
        params["W_out"]
    )

    return logits, cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # TODO: advance generation by one token using the existing KV cache and return next-token logits
    # Step 1: Embed the new token
    x = embed_tokens(
        np.array([token_id]),
        params["embedding"]
    )

    # Step 2: Calculate Q, K, V
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    # Step 3: Update KV cache
    append_kv(cache, k, v)

    # Step 4: Get all cached K and V
    K = cache["K"][:cache["length"]]
    V = cache["V"][:cache["length"]]

    # Step 5: Causal attention
    attention = causal_attention(
        q, K, V, is_causal=True
    )

    # Step 6: Output projection
    hidden = linear_projection(
        attention, params["Wo"]
    )

    # Step 7: Generate next-token logits
    logits = linear_projection(
        hidden[-1], params["W_out"]
    )

    return logits, cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    # TODO: return the number of fixed-size blocks needed to store num_tokens tokens.
    return (num_tokens + block_size - 1) // block_size

