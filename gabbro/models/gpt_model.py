import torch
import torch.nn as nn
import torch.nn.functional as F
import vector

from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        attention_dropout: float,
        max_sequence_len: int = 256,
        apply_causal_mask=True,
        post_attention_dropout=0.0,
        init_last_layer_with_zeros: bool = False,
        apply_padding_mask_fix=False,
    ):
        super().__init__()
        assert embedding_dim % n_heads == 0, "Embedding dim must be divisible by number of heads"

        self.head_dim = embedding_dim // n_heads
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.apply_causal_mask = apply_causal_mask
        self.apply_padding_mask_fix = apply_padding_mask_fix

        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Create a causal attention mask and store it as self.tril. Being a
        # buffer means that it will not be included as parameters in the model.
        self.register_buffer("tril", torch.tril(torch.ones(max_sequence_len, max_sequence_len)))
        self.dropout = nn.Dropout(attention_dropout)
        self.post_attention_dropout = nn.Dropout(post_attention_dropout)

        self.proj = nn.Linear(embedding_dim, embedding_dim)

        if init_last_layer_with_zeros:
            logger.info("Initializing weights to 0 in last linear layer of `MultiHeadAttention`")
            self.proj.weight.data.fill_(0)
            self.proj.bias.data.fill_(0)

    def forward(self, x, padding_mask=None, interaction_matrix=None):
        B, T, C = x.shape
        # x.shape (batch, time-step, channels); channels = embedding dimension
        # interaction_matrix.shape (batch, time-step, time-step, n_heads)
        # output of size (batch, time-step, embedding_dim)

        # multiply by padding mask if it's not None
        # if padding_mask is not None:
        #     original_padding_mask = padding_mask.clone()
        #     x = x * padding_mask.unsqueeze(-1)

        k = self.key(x)  # (B, T, E)
        q = self.query(x)  # (B, T, E)
        v = self.value(x)  # (B, T, E)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (B, T, E) -> (B, T, num_heads, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)

        # Transpose: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        attn_scores = q @ k.transpose(2, 3) * k.shape[-1] ** -0.5

        if padding_mask is not None:
            # save original padding mask
            original_padding_mask = padding_mask.clone()
            # (B, T) -> (B, T, T)
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, T)
            if self.apply_padding_mask_fix:
                padding_mask = padding_mask.transpose(1, 2)
            # print(f"\npadding mask: {padding_mask}")
            # (B, T, T) -> (B, n_heads, T, T)
            padding_mask = padding_mask.unsqueeze(1).expand(B, self.n_heads, T, T)
            # Need to set a finite number for the masking, instead of -inf,
            # otherwise softmax results in nans.
            # (B, n_heads, T, T)
            fill_value = float("-inf") if self.apply_padding_mask_fix else float("-1e9")
            attn_scores = attn_scores.masked_fill(padding_mask == 0, fill_value)
            # print(f"\nattn_scores: {attn_scores}")

        # Apply the causal mask, cropped to the sequence length
        # (B, n_heads, T, T)
        if self.apply_causal_mask:
            attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        if interaction_matrix is not None:
            # go from shape (B, T, T, n_heads) to (B, n_heads, T, T)
            attn_scores += interaction_matrix.transpose(1, 3)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)
        # print(f"\nattn_weights (after softmax): {attn_weights}")

        # attn_weights have shape (B, n_heads, T, T) and v (B, n_heads, T, head_dim)
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where embedding_dim = n_heads * head_dim
        context_vec = context_vec.contiguous().view(B, T, self.embedding_dim)
        context_vec = self.proj(context_vec)

        # apply dropout
        context_vec = self.post_attention_dropout(context_vec)

        # apply mask just to be sure
        if padding_mask is not None:
            context_vec = context_vec * original_padding_mask.unsqueeze(-1)

        return context_vec


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        latent_dim: int,
        attention_dropout: float,
        max_sequence_len: int = 256,
        apply_causal_mask=True,
        post_attention_dropout=0.0,
        init_last_layer_with_zeros: bool = False,
        apply_padding_mask_fix=False,
    ):
        super().__init__()
        assert embedding_dim % n_heads == 0, "Embedding dim must be divisible by number of heads"

        self.head_dim = embedding_dim // n_heads
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.apply_causal_mask = apply_causal_mask
        self.apply_padding_mask_fix = apply_padding_mask_fix

        # Compressed latent vector for keys and values, ie down projection
        self.c_kv = nn.Linear(embedding_dim, latent_dim, bias=False)
        self.c_q = nn.Linear(embedding_dim, latent_dim, bias=False)

        # Up projection
        self.k_c = nn.Linear(latent_dim, embedding_dim, bias=False)
        self.q_c = nn.Linear(latent_dim, embedding_dim, bias=False)
        self.v_c = nn.Linear(latent_dim, embedding_dim, bias=False)

        # Create a causal attention mask and store it as self.tril. Being a
        # buffer means that it will not be included as parameters in the model.
        self.register_buffer("tril", torch.tril(torch.ones(max_sequence_len, max_sequence_len)))
        self.dropout = nn.Dropout(attention_dropout)
        self.post_attention_dropout = nn.Dropout(post_attention_dropout)

        self.proj = nn.Linear(embedding_dim, embedding_dim)

        if init_last_layer_with_zeros:
            logger.info(
                "Initializing weights to 0 in last linear layer of `MultiHeadLatentAttention`"
            )
            self.proj.weight.data.fill_(0)
            self.proj.bias.data.fill_(0)

    def forward(self, x, padding_mask=None, interaction_matrix=None):
        B, T, C = x.shape
        # x.shape (batch, time-step, channels); channels = embedding dimension
        # interaction_matrix.shape (batch, time-step, time-step, n_heads)
        # output of size (batch, time-step, embedding_dim)

        k_down = self.c_kv(x)
        q_down = self.c_q(x)
        v_down = self.c_kv(x)
        k = self.k_c(k_down)  # key(x)  # (B, T, E)
        q = self.q_c(q_down)  # .query(x)  # (B, T, E)
        v = self.v_c(v_down)  # .value(x)  # (B, T, E)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (B, T, E) -> (B, T, num_heads, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)

        # Transpose: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        attn_scores = q @ k.transpose(2, 3) * k.shape[-1] ** -0.5

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, T)  # (B, T) -> (B, T, T)
            if self.apply_padding_mask_fix:
                padding_mask = padding_mask.transpose(1, 2)
            # (B, T, T) -> (B, n_heads, T, T)
            padding_mask = padding_mask.unsqueeze(1).expand(B, self.n_heads, T, T)
            # Need to set a finite number for the masking, instead of -inf,
            # otherwise softmax results in nans.
            # (B, n_heads, T, T)
            fill_value = float("-inf") if self.apply_padding_mask_fix else float("-1e9")
            attn_scores = attn_scores.masked_fill(padding_mask == 0, fill_value)

        # Apply the causal mask, cropped to the sequence length
        # (B, n_heads, T, T)
        if self.apply_causal_mask:
            attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        if interaction_matrix is not None:
            # go from shape (B, T, T, n_heads) to (B, n_heads, T, T)
            attn_scores += interaction_matrix.transpose(1, 3)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        # attn_weights have shape (B, n_heads, T, T) and v (B, n_heads, T, head_dim)
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where embedding_dim = n_heads * head_dim
        context_vec = context_vec.contiguous().view(B, T, self.embedding_dim)
        context_vec = self.proj(context_vec)

        # apply dropout
        context_vec = self.post_attention_dropout(context_vec)

        return context_vec


class FeedForward(nn.Module):
    """Simple linear layer followed by a non-linearity to be placed after the attention blocks."""

    def __init__(
        self,
        embedding_dim: int,
        factor: int = 4,
        dropout: float = 0.0,
        init_last_layer_with_zeros: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, factor * embedding_dim),
            nn.ReLU(),
            nn.Linear(factor * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        if init_last_layer_with_zeros:
            logger.info("Initializing weights to 0 in `FeedForward` net")
            self.net[-2].weight.data.fill_(0)
            self.net[-2].bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)


class GPT_DecoderBlock(nn.Module):
    """The GPT decoder block."""

    def __init__(
        self,
        embedding_dim: int,
        attention_dropout: float,
        n_heads: int,
        latent_dim: int = 0,
        verbose: bool = False,
        apply_causal_mask: bool = True,
        max_sequence_len: int = 256,
        mlp_factor=4,
        mlp_dropout=0.0,
        post_attention_dropout: float = 0.0,
        pre_norm: bool = False,
        init_identity: bool = False,
        apply_padding_mask_fix: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.apply_causal_mask = apply_causal_mask
        self.apply_padding_mask_fix = apply_padding_mask_fix
        self.pre_norm = pre_norm
        if latent_dim == 0:
            self.mha_block = MultiHeadAttention(
                embedding_dim,
                n_heads,
                attention_dropout,
                apply_causal_mask=apply_causal_mask,
                max_sequence_len=max_sequence_len,
                post_attention_dropout=post_attention_dropout,
                init_last_layer_with_zeros=init_identity,
                apply_padding_mask_fix=apply_padding_mask_fix,
            )
        else:
            self.mha_block = MultiHeadLatentAttention(  # MultiHeadAttention(
                embedding_dim,
                n_heads,
                latent_dim,
                attention_dropout,
                apply_causal_mask=apply_causal_mask,
                max_sequence_len=max_sequence_len,
                post_attention_dropout=post_attention_dropout,
                init_last_layer_with_zeros=init_identity,
                apply_padding_mask_fix=apply_padding_mask_fix,
            )
        self.ff_block = FeedForward(
            embedding_dim,
            factor=mlp_factor,
            dropout=mlp_dropout,
            init_last_layer_with_zeros=init_identity,
        )
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, padding_mask=None, interaction_matrix=None):
        if padding_mask is not None:
            padding_mask_clone = padding_mask.clone()

        x_residual = x

        # --- Multi-head attention block

        if self.pre_norm:
            x = self.layernorm_1(x)

        x = self.mha_block(x, padding_mask=padding_mask, interaction_matrix=interaction_matrix)

        x += x_residual

        if not self.pre_norm:
            x = self.layernorm_1(x)

        # --- Feed forward block

        x_residual = x

        if self.pre_norm:
            x = self.layernorm_2(x)

        x = self.ff_block(x)

        x += x_residual

        if not self.pre_norm:
            x = self.layernorm_2(x)

        if padding_mask is not None:
            x = x * padding_mask_clone.unsqueeze(-1)

        return x


class GPT_DecoderStack(nn.Module):
    """The GPT decoder stack."""

    def __init__(self, n_GPT_blocks: int, **kwargs):
        super().__init__()
        self.GPT_blocks = nn.ModuleList([GPT_DecoderBlock(**kwargs) for _ in range(n_GPT_blocks)])

    def forward(self, x, mask=None, interaction_matrix=None, **kwargs):
        for block in self.GPT_blocks:
            x = block(x, padding_mask=mask, interaction_matrix=interaction_matrix)
        return x


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        codebook_sizes,
        codebook_dim,
        embedding_kwargs={},
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, codebook_dim, **embedding_kwargs) for n in codebook_sizes]
        )

    def forward(self, x):
        y = 0
        for i, emb_table in enumerate(self.embeddings):
            y = y + emb_table(x)
        return y


class SingleEmbeddingWithMLP(nn.Module):
    """A single embedding layer with a subsequent MLP."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        output_dim,
        mlp_hidden_dim: int = 128,
        mlp_dropout: float = 0.05,
        mlp_n_layers: int = 2,
        embedding_kwargs={},
        dropout=0.0,
        noise=0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, **embedding_kwargs)
        self.noise = noise
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            *[
                nn.Sequential(
                    nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(mlp_dropout)
                )
                for _ in range(mlp_n_layers - 1)
            ],
            nn.Linear(mlp_hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        x = self.mlp(x)
        return x
