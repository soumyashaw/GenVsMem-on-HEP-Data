"""Collection of the different functional forms of attention.
From: https://github.com/mattcleigh/mltools
"""

import math

import torch as T
import torch.nn.functional as F
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from torch import nn


def apply_rope(x: T.Tensor, freqs_cis: T.Tensor) -> T.Tensor:
    """Rotate the input tensor using the precomputed frequencies."""
    out = x.reshape(*x.shape[:-1], x.shape[-1] // 2)
    out = T.view_as_complex(out.float())
    out = T.view_as_real(out * freqs_cis)
    return out.view_as(x).type_as(x)


def merge_masks(
    kv_mask: T.BoolTensor | None,
    attn_mask: T.BoolTensor | None,
    attn_bias: T.Tensor | None,
    query: T.Tensor,
    causal: bool = False,
) -> T.BoolTensor | None:
    """Create a full attention mask which uses the padding information and bias."""
    # Create the placeholder for the full mask, None is full attention
    merged_mask = None

    # If the kv_mask mask exists, we ensure that padded tokens never send information
    if kv_mask is not None:
        merged_mask = kv_mask.unsqueeze(-2).expand(-1, query.shape[-2], -1)

    # If attention mask exists, combine it with the existing
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    # If using a causal mask, then set the upper triangle to False
    # print(f"Mask before causal: {merged_mask}")
    if causal and merged_mask is not None:
        merged_mask = merged_mask.tril()

    # print(f"Mask after causal: {merged_mask}")

    # print(f"merged_mask: {merged_mask}")

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    # If the attention bias exists, convert to a float and add to the mask
    if attn_bias is not None:
        if merged_mask is not None:
            merged_mask = T.where(merged_mask, 0, -T.inf).type(query.dtype)
            merged_mask = merged_mask + attn_bias.permute(0, 3, 1, 2)
        else:
            merged_mask = attn_bias.permute(0, 3, 1, 2)

    return merged_mask


def standard_attention(
    x: T.Tensor,
    *,
    num_heads: int,
    linear: nn.Linear,
    kv: T.Tensor | None = None,
    mask: T.Tensor | None = None,
    kv_mask: T.Tensor | None = None,
    attn_mask: T.Tensor | None = None,
    attn_bias: T.Tensor | None = None,
    drop: float = 0.0,
    causal: bool = False,
    qk_norm: nn.Module | None,
    rope_freqs: T.Tensor | None = None,
    kv_rope_freqs: T.Tensor | None = None,
    **_,  # To allow for consistent API
) -> T.Tensor:
    """Standard multi-head attention function using pytorch backend."""
    B, S, D = x.shape
    HD = D // num_heads
    NH = num_heads

    # Self-Attention: Single operation
    if kv is None:
        q, k, v = linear(x).reshape(B, S, 3, NH, HD).permute(2, 0, 3, 1, 4).unbind(0)

    # Cross-Attention: Separate projections and reshaping
    else:
        weight, bias = linear.weight, linear.bias
        w_q, w_kv = weight.split([D, D * 2])  # split is a view (very cheap)
        b_q, b_kv = bias.split([D, D * 2]) if bias is not None else (None, None)

        # do separate projections
        q = F.linear(x, w_q, b_q)
        k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)

        # reshaping -> B,NH,S,HD
        q = q.view(B, -1, NH, HD).transpose(1, 2).contiguous()
        k = k.view(B, -1, NH, HD).transpose(1, 2).contiguous()
        v = v.view(B, -1, NH, HD).transpose(1, 2).contiguous()

    # Apply the optional operations on the query and key tensors
    if rope_freqs is not None:
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, kv_rope_freqs if kv_rope_freqs is not None else rope_freqs)
    if qk_norm is not None:
        q, k = qk_norm(q, k)

    # print(f"causal: {causal}")

    # run attention -> B,NH,S,HD
    kv_mask = mask if kv is None else kv_mask  # who is sending? CA or SA?
    a_mask = merge_masks(kv_mask, attn_mask, attn_bias, q, causal)
    c = causal and a_mask is None  # a_mask will at least incl causal
    a_out = F.scaled_dot_product_attention(q, k, v, a_mask, drop, is_causal=c)

    # recombine heads -> B,S,D
    return a_out.transpose(1, 2).contiguous().view(B, S, D)


def flash_self_attention(
    x: T.Tensor,
    *,
    culens: T.Tensor,
    maxlen: int,
    num_heads: int,
    linear: nn.Linear,
    drop: float = 0.0,
    causal: bool = False,
    qk_norm: nn.Module | None = None,
    **_,  # To allow for consistent API
) -> T.Tensor:
    """Optimized self-attention for variable sized sets using the flash backend."""
    BS, D = x.shape
    qkv = linear(x).view(-1, 3, num_heads, D // num_heads)
    if qk_norm is None:
        attn = flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, drop, causal=causal)
    else:
        q, k, v = qkv.unbind(1)
        q, v = qk_norm(q, v)
        attn = flash_attn_varlen_func(q, k, v, culens, culens, maxlen, maxlen, drop, causal=causal)
    return attn.contiguous().view(BS, D)


def flash_cross_attention(
    x: T.Tensor,
    *,
    culens: T.Tensor,
    maxlen: int,
    kv: T.Tensor,
    kv_culens: T.Tensor,
    kv_maxlen: int,
    num_heads: int,
    linear: nn.Linear,
    drop: float = 0.0,
    qk_norm: nn.Module | None = None,
    **_,  # To allow for consistent API
) -> T.Tensor:
    """Optimized cross-attention for variable sized sets using the flash backend."""
    BS, D = x.shape
    HD = D // num_heads
    weight, bias = linear.weight, linear.bias
    w_q, w_kv = weight.split([D, D * 2])
    b_q, b_kv = bias.split([D, D * 2]) if bias is not None else (None, None)
    q = F.linear(x, w_q, b_q).view(-1, num_heads, HD)
    kv = F.linear(kv, w_kv, b_kv).view(-1, 2, num_heads, HD)

    if qk_norm is None:
        attn = flash_attn_varlen_kvpacked_func(q, kv, culens, kv_culens, maxlen, kv_maxlen, drop)
    else:
        k, v = kv.unbind(1)
        q, k = qk_norm(q, k)
        attn = flash_attn_varlen_func(q, k, v, culens, kv_culens, maxlen, kv_maxlen, drop)
    return attn.contiguous().view(BS, D)


def my_scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: T.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    pad_val: float = -float("inf"),
) -> T.Tensor:
    """Compute dot product attention using the given query, key, and value tensors.

    Deprecated! Was originally used for ONNX exports, but
    the standard nn.functional.scaled_dot_product_attention has been supported by ONNX
    since opset 14!

    Parameters
    ----------
    query : T.Tensor
        The query tensor.
    key : T.Tensor
        The key tensor.
    value : T.Tensor
        The value tensor.
    attn_mask : T.Tensor | None, optional
        The attention mask tensor, by default None.
    dropout_p : float, optional
        The dropout probability, by default 0.0.
    is_causal : bool, optional
        Whether to use causal attention, by default False.
    scale: float | None, optional
        The scale factor to divide the attention weights by, by default None.
    pad_val : float, optional
        The padding value for the attention mask, by default -float("inf").

    Returns
    -------
    T.Tensor
        The result of the scaled dot product attention operation.
    """
    # Get the shapes and set the scale
    QS = query.shape[-2]
    KS = key.shape[-2]
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Build the attention bias as a float
    attn_bias = T.zeros(QS, KS, dtype=query.dtype, device=query.device)

    # If using a causal mask, then set the upper triangle to the pad value
    if is_causal:
        assert attn_mask is None, "Causal attention does not support attention masks!"
        attn_mask = T.ones(QS, KS, dtype=T.bool).tril(diagonal=0)
        attn_bias.masked_fill_(~attn_mask, pad_val)

    # If proved own attention mask, then add it to the bias
    elif attn_mask is not None:
        if attn_mask.dtype == T.bool:
            attn_bias.masked_fill_(~attn_mask, pad_val)
        else:
            attn_bias += attn_mask

    # Apply the attention operation using the mask as a bias
    attn_weight = query @ key.transpose(-2, -1) * scale
    attn_weight = F.softmax(attn_weight + attn_bias, dim=-1)
    attn_weight = T.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value
