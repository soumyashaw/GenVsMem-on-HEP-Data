"""Some classes to describe transformer architectures.
From: https://github.com/mattcleigh/mltools
"""

import logging
from typing import Any

import torch as T
import torch.nn.functional as F
from torch import nn

from .attention import (
    flash_cross_attention,
    flash_self_attention,
    standard_attention,
)
from .torch_utils import ParameterNoWD, append_dims

log = logging.getLogger(__name__)

# A list of all the keyword arguments that are supported by the attention functions
ATTN_KWARGS = [
    "ctxt",
    "kv",
    "mask",
    "kv_mask",
    "attn_mask",
    "attn_bias",
    "causal",
    "rope_freqs",
    "kv_rope_freqs",
    "culens",
    "maxlen",
    "kv_culens",
    "kv_maxlen",
]


def pos_embed(embed_dim: int, max_seq_len: int):
    """Create the positional embedding for the transformer."""
    assert embed_dim % 2 == 0

    # Create the increasing frequencies for the sin and cos functions
    omega = T.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    # Get the positions from the max sequence length
    pos = T.arange(max_seq_len, dtype=float).reshape(-1)

    # Create the matrix using the outer product of the positions and frequencies
    out = omega.unsqueeze(0) * pos.unsqueeze(-1)  # (S, D/2)

    # Embed using sin and cos functions then combine
    emb_sin = T.sin(out)  # (S, D/2)
    emb_cos = T.cos(out)  # (S, D/2)
    pos_emb = T.cat([emb_sin, emb_cos], axis=1)  # (S, D)

    return pos_emb.unsqueeze(0).float()  # For batch dimension


def calc_rope_freqs(x: T.Tensor, num_heads: int, theta: float = 10000.0) -> T.Tensor:
    """Precompute the frequencies for the rotary positional encoding."""
    _B, S, D = x.shape
    HD = D // num_heads
    freqs = 1.0 / (theta ** (T.arange(0, HD, 2, device=x.device).float() / HD))
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = T.outer(t, freqs)
    return T.polar(T.ones_like(freqs), freqs)


def pack(
    x: T.Tensor,
    mask: T.BoolTensor,
    ctxt: T.Tensor | None = None,
):
    """Undo all padding and compress the sequence."""
    assert mask is not None, "Mask is required to pack the sequence!"

    # Get the culens and maxlen variables needed by the flash attention func
    seqlens = mask.sum(dim=-1)
    culens = F.pad(T.cumsum(seqlens, dim=-1), (1, 0), value=0).int()
    maxlen = seqlens.max().item()

    # Context info gets tricky because it may need to be repeated
    if ctxt is not None:
        if (dim_diff := x.dim() - ctxt.dim()) > 0:  # Expand then pack (no mem copy)
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*x.shape[:-1], -1)
        ctxt = ctxt[mask]

    # Replace x with its compressed version
    return x[mask], culens, maxlen, ctxt


def unpack(x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Take a compressed sequence and unpack it to a padded tensor."""
    out = T.zeros((*mask.shape, x.shape[-1]), dtype=x.dtype, device=x.device)
    out[mask] = x
    return out


def add_registers(
    x: T.Tensor,
    reg: T.Tensor,
    add_to_both: bool = False,
    **kwargs,
) -> tuple:
    """Add registers to the front of the input and accommodate the mask.

    add_to_both indicates whether to modify the attn_mask and bias at both the recv
    and send dimensions. This is primarily because the encoder and decoder use
    these differently. In the encoder the attn mask is between the kv and x tensors
    while in the decoder the attn mask is between the x and x tensors.
    """
    # expand the registers so they can be broadcasted for the whole batch
    reg = reg.expand(x.size(0), -1, x.shape[-1])
    nreg = reg.shape[1]

    # add the registers to the FRONT of the input
    x = T.cat([reg, x], dim=-2)  # Sequence dimension

    # Add the mask for the registers with trues at the front
    mask = kwargs.get("mask")
    if mask is not None:
        mask = F.pad(mask, (nreg, 0), value=True)
        kwargs["mask"] = mask

    # Add the attention mask for the registers
    # The attention mask is b x recv x send
    # We are adding to the recv dimension ONLY!!!
    attn_mask = kwargs.get("attn_mask")
    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, (nreg * add_to_both, 0, nreg, 0), value=True)
        kwargs["attn_mask"] = attn_mask

    # Add an attention bias of zero for the registers
    attn_bias = kwargs.get("attn_bias")
    if attn_bias is not None:
        attn_bias = F.pad(attn_bias, (0, 0, nreg * add_to_both, 0, nreg, 0), value=0)
        kwargs["attn_bias"] = attn_bias

    return x, kwargs


class Identity(nn.Module):
    """Like nn.Identity but with kwargs in forward method."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    def forward(self, x, **_kwargs) -> Any:
        return x


class QKNorm(nn.Module):
    """Wrap both the query and key normalisation layers."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(dim, elementwise_affine=False)

    def forward(self, q: T.Tensor, k: T.Tensor) -> tuple:
        return self.q_norm(q).type(q.dtype), self.k_norm(k).type(q.dtype)


class Residual(nn.Module):
    """Wraps a module with a normalisation layer, residual connection, and gating.

    If context is provided, it is used for adaptive normalisation and gating.
    Gating is always initialised as zero, so the module is initially bypassed.
    """

    def __init__(self, fn: nn.Module, ctxt_dim: int = 0) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
            Used in the modulator to determine the scale, shift and gate.
        """
        super().__init__()
        assert hasattr(fn, "dim"), "Module in residual layer must have a dim attribute!"
        self.dim = fn.dim
        self.fn = fn
        self.ctxt_dim = ctxt_dim
        self.norm = nn.RMSNorm(self.dim, elementwise_affine=False)
        if ctxt_dim:
            self.scale = nn.Linear(ctxt_dim, self.dim)  # Separate as its easier to log
            self.shift = nn.Linear(ctxt_dim, self.dim)
            self.gate = nn.Linear(ctxt_dim, self.dim)
        else:
            self.gate = nn.Parameter(T.ones(self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ctxt_dim:
            self.scale.weight.data.zero_()
            self.scale.bias.data.zero_()
            self.shift.weight.data.zero_()
            self.shift.bias.data.zero_()
            self.gate.weight.data.zero_()
            self.gate.bias.data.zero_()
        else:
            self.gate.data.zero_()

    def __repr__(self) -> str:
        return f"Res-{self.fn}"

    def forward(
        self,
        x: T.Tensor,
        *args,
        ctxt: T.Tensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        if self.ctxt_dim:
            assert ctxt is not None, f"{self} initialised with ctxt_dim but none given!"
            ctxt = F.silu(ctxt)
            scale = append_dims(self.scale(ctxt), x.dim(), dim=1)
            shift = append_dims(self.shift(ctxt), x.dim(), dim=1)
            gate = append_dims(self.gate(ctxt), x.dim(), dim=1)
            tmp = self.norm(x) * (scale + 1) + shift
            return x + self.fn(tmp, *args, **kwargs) * gate
        return x + self.fn(self.norm(x), *args, **kwargs) * self.gate


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim  # Useful for when wrapping the module in residual
        self.lin1 = nn.Linear(dim, 2 * mult * dim)
        self.lin2 = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class Attention(nn.Module):
    """Basic multiheaded attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0,
        do_qknorm: bool = True,
    ) -> None:
        """Initialise the attention block.

        Parameters
        ----------
        dim : int
            The dimension of the input and output.
        num_heads : int, optional
            The number of attention heads, by default 1.
        dropout : float, optional
            The dropout probability, by default 0.
        do_scale : bool, optional
            Have a trainable scale for the attention operation, by default True.
        do_qknorm : bool, optional
            Whether to use RMSNorm on the query and key, by default False.
        """
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by the number of heads!"
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.do_qknorm = do_qknorm
        self.attn_in = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.final_norm = nn.RMSNorm(dim)
        self.qk_norm = QKNorm(dim // num_heads) if do_qknorm else None

    def _get_attn_fn(self, **kwargs) -> tuple:
        """Work out which attention function to use based on the inputs."""
        has_culens = kwargs.get("culens") is not None
        has_maxlen = kwargs.get("maxlen") is not None
        has_kv = kwargs.get("kv") is not None
        if has_culens and has_maxlen:
            has_ropes = kwargs.get("rope_freqs") is not None
            attn_mask = kwargs.get("attn_mask") is not None
            attn_bias = kwargs.get("attn_bias") is not None
            assert not attn_mask, "Packed attn does not support attention masks!"
            assert not attn_bias, "Packed attn does not support attention bias!"
            assert not has_ropes, "Packed attn does not support rotary encoding!"
            if has_kv:
                return flash_cross_attention
            return flash_self_attention
        return standard_attention

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Dispatch to the appropriate attention function based on the inputs."""
        attn_fn = self._get_attn_fn(**kwargs)
        # print(f"Using {attn_fn.__name__} for attention")
        a_out = attn_fn(
            x,
            linear=self.attn_in,
            num_heads=self.num_heads,
            drop=self.dropout if self.training else 0.0,
            qk_norm=self.qk_norm,
            **kwargs,
        )
        a_out = self.final_norm(a_out)
        return self.attn_out(a_out)


class EncoderBlock(nn.Module):
    """Building block for the Transformer Encoder containing MHSA and FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_config: dict | None = None,
        attn_config: dict | None = None,
    ) -> None:
        """Initialise the encoder block.

        Parameters
        ----------
        dim : int
            The dimension of of the block
        ctxt_dim : int, optional
            The dimension of the context, by default 0
            Used in the residual modulators to determine the scale, shift and gate.
        ff_config : dict, optional
            The keyword arguments for the feedforward network, by default None
        attn_config : dict, optional
            The keyword arguments for the attention block, by default None
        """
        super().__init__()
        self.dim = dim
        attn_config = attn_config or {}
        ff_config = ff_config or {}
        self.sa = Residual(Attention(dim, **attn_config), ctxt_dim)
        self.ff = Residual(SwiGLUNet(dim, **ff_config), ctxt_dim)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None, **kwargs) -> T.Tensor:
        x = self.sa(x, ctxt=ctxt, **kwargs)
        return self.ff(x, ctxt=ctxt)


class DecoderBlock(nn.Module):
    """Building block for the Transformer Decoder containing SA+CA+FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_config: dict | None = None,
        attn_config: dict | None = None,
        ca_first: bool = False,
    ) -> None:
        """Initialise the encoder block.

        Parameters
        ----------
        dim : int
            The dimension of of the block
        ctxt_dim : int, optional
            The dimension of the context, by default 0
            Used in the residual modulators to determine the scale, shift and gate.
        ff_config : dict, optional
            The keyword arguments for the feedforward network, by default None
        attn_config : dict, optional
            The keyword arguments for the attention block, by default None
        ca_first : bool, optional
            Whether to do the cross attention before the self attention, by default
            False
        """
        super().__init__()
        self.dim = dim
        attn_config = attn_config or {}
        ff_config = ff_config or {}
        self.ca_first = ca_first
        self.sa = Residual(Attention(dim, **attn_config), ctxt_dim)
        self.ca = Residual(Attention(dim, **attn_config), ctxt_dim)
        self.ff = Residual(SwiGLUNet(dim, **ff_config), ctxt_dim)

    def forward(
        self,
        x: T.Tensor,
        *,
        kv: T.Tensor,
        ctxt: T.Tensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        """Pass through the decoder block."""
        ca_kwargs = {"kv": kv, "attn_mask": None, "attn_bias": None}
        sa_kwargs = {"kv": None, "attn_mask": attn_mask, "attn_bias": attn_bias}
        if self.ca_first:
            x = self.ca(x, ctxt=ctxt, **ca_kwargs, **kwargs)
            x = self.sa(x, ctxt=ctxt, **sa_kwargs, **kwargs)
        else:
            x = self.sa(x, ctxt=ctxt, **sa_kwargs, **kwargs)
            x = self.ca(x, ctxt=ctxt, **ca_kwargs, **kwargs)
        return self.ff(x, ctxt=ctxt)


class Transformer(nn.Module):
    """Simple transformer stack of encoder or decoder blocks.

    Includes option to add registers from: doi.org/10.48550/arXiv.2309.16588.
    Single register can be thought of as the class token.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        inpt_dim: int = 0,
        outp_dim: int = 0,
        num_layers: int = 6,
        max_seq_len: int = 0,
        num_registers: int = 0,
        pos_enc: str = "none",
        use_decoder: bool = False,
        pack_inputs: bool = False,
        unpack_output: bool = True,
        layer_config: dict | None = None,
    ) -> None:
        """Parameters
        ----------
        dim : int, optional
            Dimension of the model (default is 128).
        ctxt_dim : int, optional
            Context dimension (default is 0).
        inpt_dim : int, optional
            Input dimension (default is 0).
        outp_dim : int, optional
            Output dimension (default is 0).
        num_layers : int, optional
            Number of layers in the transformer (default is 6).
        max_seq_len : int, optional
            Maximum sequence length for absolute encoding (default is 0).
        num_registers : int, optional
            Number of registers (default is 0).
        pos_enc : str, optional
            Type of positional encoding to use (default is "none").
        use_decoder : bool, optional
            Whether to use the transformer as a decoder (default is False).
        pack_inputs : bool, optional
            Whether to pack inputs (default is False).
        unpack_output : bool, optional
            Whether to unpack output (default is True).
        layer_config : dict, optional
            Configuration dictionary for the layers (default is None).
        """
        super().__init__()
        assert pos_enc in {"none", "abs", "rotary"}, "Invalid positional encoding!"
        assert not (pos_enc == "abs" and max_seq_len == 0), "Maxlen required for abs!"
        layer_config = layer_config or {}
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.inpt_dim = inpt_dim or dim
        self.outp_dim = outp_dim or dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_registers = num_registers
        self.pos_enc = pos_enc
        self.use_decoder = use_decoder
        self.pack_inputs = pack_inputs
        self.unpack_output = unpack_output

        # Base repeated transformer layers
        layer = DecoderBlock if use_decoder else EncoderBlock
        self.layers = nn.ModuleList(
            [layer(dim, ctxt_dim, **layer_config) for _ in range(num_layers)]
        )
        self.num_heads = self.layers[0].sa.fn.num_heads

        # Output norm and linear layer
        self.final_norm = nn.RMSNorm(dim)
        self.linear_out = nn.Linear(dim, self.outp_dim)

        # Optional features
        if self.inpt_dim != self.dim:
            self.linear_in = nn.Linear(self.inpt_dim, self.dim)
        if self.num_registers:
            self.registers = ParameterNoWD(T.randn((self.num_registers, dim)) * 1e-3)
        if self.pos_enc == "abs":
            self.abs_enc = ParameterNoWD(T.randn((max_seq_len, dim)) * 1e-3)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass through all layers of the transformer."""
        assert not (self.num_registers and "culens" in kwargs), (
            "Cannot add registers to inputs which are already packed!"
        )

        # Project the inputs if there is a size mismatch
        if self.inpt_dim != self.dim:
            x = self.linear_in(x)

        # Add absolute positional encoding information
        if self.pos_enc == "abs":
            x = x + self.abs_enc[: x.shape[-2], :]  # Trimmed to seq len

        # Add registers to the FRONT of the input
        if self.num_registers:
            x, kwargs = add_registers(x, self.registers, **kwargs, add_to_both=self.use_decoder)

        # Add rotary positional encoding information (had to be after the registers)
        if self.pos_enc == "rotary":
            kwargs["rope_freqs"] = calc_rope_freqs(x, self.num_heads)
            kv = kwargs.get("kv")
            if kv is not None:
                kwargs["kv_rope_freqs"] = calc_rope_freqs(kv, self.num_heads)

        # Compress all inputs for the flash attention function
        # This also adds the needed kwargs
        if self.pack_inputs and "culens" not in kwargs:
            x, culens, maxlen, ctxt = pack(x, kwargs.get("mask"), kwargs.get("ctxt"))
            kwargs.update(culens=culens, maxlen=maxlen, ctxt=ctxt)
            kv = kwargs.get("kv")
            if kv is not None and "kv_culens" not in kwargs:
                kv, kv_culens, kv_maxlen, _ = pack(kv, kwargs.get("kv_mask"))
                kwargs.update(kv_culens=kv_culens, kv_maxlen=kv_maxlen)

        # Now we can actually pass through each layer
        for layer in self.layers:
            x = layer(x, **kwargs)
        x = self.final_norm(x)
        x = self.linear_out(x)

        # Decompress the output if it was packed
        if "culens" in kwargs:
            if self.unpack_output:
                return unpack(x, kwargs["mask"])
            return x, kwargs["culens"], kwargs["maxlen"]
        return x  # Otherwise just return x

    def remove_registers(self, x: T.Tensor) -> tuple:
        """Remove the registers from the front of the input."""
        return x[:, : self.num_registers], x[:, self.num_registers :]

    def get_combined_mask(self, mask: T.BoolTensor | None) -> T.BoolTensor | None:
        """Get a mask which can be used for the combined register+sequence tensor."""
        if self.num_registers == 0:
            return mask
        if mask is None:
            return None
        return F.pad(mask, (self.num_registers, 0), value=True)


class CrossAttentionEncoder(nn.Module):
    """Lopsided transformer which upades two point clouds x1, x2.

    Only applied self-attention to the second point cloud.
    Computationally efficient if N(x2) << N(x1).
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 2,
        enc_config: dict | None = None,
        dec_config: dict | None = None,
        pack_inputs: bool = False,
        unpack_output: bool = True,
    ) -> None:
        super().__init__()
        enc_config = enc_config or {}
        dec_config = dec_config or {}
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_layers = num_layers
        self.pack_inputs = pack_inputs
        self.unpack_output = unpack_output

        # Each layer needs a decoder block and an encoder block
        self.dec_layers = nn.ModuleList(
            [DecoderBlock(dim, ctxt_dim, **dec_config) for _ in range(num_layers)]
        )
        self.enc_layers = nn.ModuleList(
            [EncoderBlock(dim, ctxt_dim, **enc_config) for _ in range(num_layers)]
        )
        self.x1_final_norm = nn.RMSNorm(dim)
        self.x2_final_norm = nn.RMSNorm(dim)

    def forward(
        self,
        x1: T.Tensor,
        x2: T.Tensor,
        x1_mask: T.BoolTensor | None = None,
        x2_mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        x2_attn_mask: T.BoolTensor | None = None,
        x2_attn_bias: T.Tensor | None = None,
        x2_causal: bool = False,
    ) -> T.Tensor:
        """Pass through all layers of the transformer."""
        if self.pack_inputs:
            x1, x1_culens, x1_maxlen, x1_ctxt = pack(x1, x1_mask, ctxt)
            x2, x2_culens, x2_maxlen, x2_ctxt = pack(x2, x2_mask, ctxt)
        else:
            x1_culens = x1_maxlen = x2_culens = x2_maxlen = None
            x1_ctxt = x2_ctxt = ctxt

        for dec_layer, enc_layer in zip(self.dec_layers, self.enc_layers, strict=True):
            x2 = dec_layer(
                x=x2,
                kv=x1,
                ctxt=x2_ctxt,
                mask=x2_mask,
                kv_mask=x1_mask,
                attn_mask=x2_attn_mask,
                attn_bias=x2_attn_bias,
                culens=x2_culens,
                maxlen=x2_maxlen,
                kv_culens=x1_culens,
                kv_maxlen=x1_maxlen,
                causal=x2_causal,
            )
            x1 = enc_layer(
                x=x1,
                kv=x2,
                ctxt=x1_ctxt,
                mask=x1_mask,
                kv_mask=x2_mask,
                culens=x1_culens,
                maxlen=x1_maxlen,
                kv_culens=x2_culens,
                kv_maxlen=x2_maxlen,
            )
        x1 = self.x1_final_norm(x1)
        x2 = self.x2_final_norm(x2)
        if self.pack_inputs:
            if self.unpack_output:
                return unpack(x1, x1_mask), unpack(x2, x2_mask)
            return (x1, x1_culens, x1_maxlen), (x2, x2_culens, x2_maxlen)
        return x1, x2


class ClassAttentionPooling(nn.Module):
    """Pooling operation that uses attention."""

    def __init__(
        self,
        *,
        dim: int = 128,
        inpt_dim: int = 0,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        num_layers: int = 1,
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        layer_config = layer_config or {}
        self.dim = dim
        self.inpt_dim = inpt_dim or dim
        self.outp_dim = outp_dim or dim
        self.ctxt_dim = ctxt_dim
        self.num_layers = num_layers

        # The single trainable global token
        self.global_token = nn.Parameter(T.randn((1, 1, self.dim)))

        # Main cross attention layers
        self.layers = nn.ModuleList(
            [EncoderBlock(self.dim, ctxt_dim, **layer_config) for _ in range(num_layers)]
        )

        # Extra layers
        self.linear_in = nn.Linear(self.inpt_dim, dim)
        self.final_norm = nn.RMSNorm(dim)
        self.linear_out = nn.Linear(dim, self.outp_dim)

    def forward(self, x: T.Tensor, mask: T.BoolTensor | None = None, **kwargs) -> T.Tensor:
        """Perform class attention pooling on a sequence."""
        x = self.linear_in(x)
        x = F.silu(x)

        # If x is packed, then so too must be the global token
        if "culens" in kwargs:
            culens = kwargs["culens"]
            maxlen = kwargs["maxlen"]
            B = culens.size(0) - 1
            g = self.global_token.squeeze(1).expand(B, self.dim)
            kwargs["culens"] = T.arange(B + 1, device=culens.device, dtype=culens.dtype)
            kwargs["maxlen"] = 1
            kwargs["kv_culens"] = culens
            kwargs["kv_maxlen"] = maxlen

        # Otherwise we broadcast the global token to match the batch size
        else:
            g = self.global_token.expand(x.shape[0], 1, self.dim)

        # Pass through the layers
        for layer in self.layers:
            g = layer(g, kv=x, kv_mask=mask, **kwargs)
        g = self.final_norm(g)
        g = self.linear_out(g)

        # If not packed, then we pop out the sequence dimension
        # If packed, then the format is already correct
        if "culens" not in kwargs:
            g.squeeze_(-2)
        return g
