"""Decode integral token IDs emitted through float-typed evaluation tensors."""

from __future__ import annotations


def cast_integral_token_ids(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [cast_integral_token_ids(item) for item in value]

    numeric = float(value)
    if not numeric.is_integer():
        raise TypeError(f"Non-integral token ID: {value!r}")
    return int(numeric)


def install_integral_token_id_decode(tokenizer):
    """Wrap one tokenizer instance without changing upstream evaluation code."""
    if getattr(tokenizer, "_cegr_v2_integral_decode", False):
        return tokenizer

    original_decode = tokenizer.decode
    original_batch_decode = tokenizer.batch_decode

    def decode(token_ids, *args, **kwargs):
        return original_decode(cast_integral_token_ids(token_ids), *args, **kwargs)

    def batch_decode(token_ids, *args, **kwargs):
        return original_batch_decode(
            cast_integral_token_ids(token_ids), *args, **kwargs
        )

    tokenizer.decode = decode
    tokenizer.batch_decode = batch_decode
    tokenizer._cegr_v2_integral_decode = True
    return tokenizer
