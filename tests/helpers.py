"""Small models shared by the tests (import as ``from helpers import gpt2``)."""

import torch
import transformers


def gpt2(dtype=None, *, dropout=True, **overrides):
    """A 2-layer GPT-2 (64 wide, vocabulary 500, tied embeddings) seeded with 0; ``dropout=False``
    sets every dropout to 0; ``overrides`` go to ``GPT2Config``."""
    torch.manual_seed(0)
    config = {"n_layer": 2, "n_embd": 64, "n_head": 2, "vocab_size": 500, "n_positions": 64}
    if not dropout:
        config |= {"attn_pdrop": 0.0, "resid_pdrop": 0.0, "embd_pdrop": 0.0}
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(**config | overrides))
    return model if dtype is None else model.to(dtype)
