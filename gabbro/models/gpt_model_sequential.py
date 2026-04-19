"""Simple GPT model.

Based on https://github.com/karpathy/ng-video-lecture.
"""

import time

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(21190)

verbose = False


class Head(nn.Module):
    """One head of self-attention Padding mask is either None or a (B,T)-size tensor of float
    values of 0 and 1."""

    def __init__(self, head_size, embedding_dim, attention_dropout):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        # Create a lower triangular matrix and stores it as self.tril. Being
        # a buffer means that it will not be included as parameters in the
        # model.
        self.register_buffer("tril", torch.tril(torch.ones(embedding_dim, embedding_dim)))
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, padding_mask=None):
        # input of size (batch, time-step, channels) channels = embedding dimension
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs) hs = head size/head dimension = embedding_dim // n_heads
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        weights = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if torch.isnan(weights).any():
            print("NaN detected after initial Q*K weight creation")
        # Apply padding mask
        if padding_mask is not None:
            # Reshape the padding mask to match dimensions with the input. The
            # original padding mask has shape (B, T), and consists of float values
            # of 0 and 1.
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, T)  # (B, T, T)
            # Need to set a finite number for the masking, instead of -inf,
            # otherwise softmax results in nans. For full precision (float32),
            # we can do -1e9, but for half precision (float16) we can at most do
            # -1e4. It needs to be a small value, so the result will be
            # negligible after softmax.
            # weights = weights.masked_fill(padding_mask == 0, float('-1e4')) # (B, T, T)
            weights = weights.masked_fill(padding_mask == 0, float("-1e9"))  # (B, T, T)
            if torch.isnan(weights).any():
                print("NaN detected after padding mask")
        # Apply causal mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        if torch.isnan(weights).any():
            print("NaN detected after causal mask")
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        if torch.isnan(weights).any():
            print("NaN detected after softmax")
            print("Number of nans after softmax: ", len(torch.isnan(weights)))
            print("Indices of nans after softmax: ", torch.argwhere(torch.isnan(weights)))
        weights = self.dropout(weights)
        if torch.isnan(weights).any():
            print("NaN detected after dropout")
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        if torch.isnan(v).any():
            print("NaN detected in v")
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        if torch.isnan(out).any():
            print("NaN detected after multiplication with V")
            print("After multiplication with V: out[5,5,:5]: ", out[5, 5, :5])
            print("V[5,5,:5]: ", v[5, 5, :5])
        return out


class MultiHeadAttentionBlock(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, embedding_dim, attention_dropout, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embedding_dim, attention_dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(head_size * n_heads, embedding_dim)
        # Need to project to original embedding dimension, otherwise we can't
        # add the residuals
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, padding_mask=None):
        out = torch.cat([h(x, padding_mask=padding_mask) for h in self.heads], dim=-1)
        # if verbose:
        #    print('out shape 1 in MHA block: ', out.shape)
        out = self.dropout(self.proj(out))
        # if verbose:
        #    print('out shape 2 in MHA block: ', out.shape)
        #
        #    print(f'Regular attention timer: {end_time-init_time}')
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity To be placed after the attention
    blocks."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GPT_DecoderBlock(nn.Module):
    """The GPT decoder block."""

    def __init__(self, embedding_dim, attention_dropout, n_heads, verbose=False):
        super().__init__()
        self.verbose = verbose
        # if self.verbose:
        #     print('Verbosity turned on in GPT_DecoderBlock')
        # else:
        #     print('Verbosity not turned on in GPT_DecoderBlock')
        head_size = embedding_dim // n_heads
        self.mha_block = MultiHeadAttentionBlock(
            embedding_dim, attention_dropout, head_size, n_heads
        )
        # self.ff_block = FeedForwardBlock(
        #    embedding_dim, embedding_dim, n_linear_layers=n_linear_layers
        #    )
        self.ff_block = FeedForward(embedding_dim)
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, padding_mask=None):
        x_residual = x
        # time_before_mha = time.time()
        x = self.mha_block(x, padding_mask=padding_mask)
        # if self.verbose:
        #    print(f'Time for MHA block: {time.time()-time_before_mha}')
        # print('x shape after MHA: ', x.shape)
        x += x_residual

        # time_before_LN = time.time()
        x = self.layernorm_1(x)
        # if self.verbose:
        #    print(f'Time for LN1: {time.time()-time_before_LN}')
        x_residual = x
        # time_before_FF = time.time()
        x = self.ff_block(x)
        # if self.verbose:
        #    print(f'Time for FF block: {time.time()-time_before_FF}')
        # print('x shape after FF: ', x.shape)
        x += x_residual

        # time_before_LN = time.time()
        x = self.layernorm_2(x)
        # if self.verbose:
        #    print(f'Time for LN2: {time.time()-time_before_LN}')
        # print('x shape after layer norm: ', x.shape)

        return x


class FullModel(nn.Module):
    """Combining the embedding, GPT decoder block, and final output architecture."""

    def __init__(
        self,
        embedding_dim,
        attention_dropout,
        vocab_size,
        max_sequence_len,
        n_heads,
        n_GPT_blocks,
        n_classes=2,
        classify=False,
        verbosity=True,
        return_embeddings=False,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.verbose = verbosity
        self.return_embeddings = return_embeddings
        # if self.verbose:
        #    print('Verbosity turned on in FullModel')
        # else:
        #    print('Verbosity not turned on in FullModel')
        self.classify = classify

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)  # , sparse=True)

        GPT_block_stack = []
        for _ in range(n_GPT_blocks):
            GPT_block_stack.extend(
                [
                    GPT_DecoderBlock(
                        embedding_dim,
                        attention_dropout,
                        n_heads=n_heads,
                        verbose=self.verbose,
                    )
                ]
            )
        self.GPT_blocks = nn.Sequential(*GPT_block_stack)

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        self.classification_head = nn.Linear(vocab_size, n_classes)

    def generate_output(self, idx):
        """Generate jet constituents autoregressively."""
        # idx is (B, T) array of indices in the current context
        for i in range(self.max_sequence_len):
            # get the predictions
            logits = self(idx)
            # print('Logit shape input for generation: ', logits.shape) if self.verbose else None
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # If it for some reason generates the start token, redo the generation:
            while idx_next == 0:
                idx_next = torch.multinomial(probs, num_samples=1)
            # print('idx_next shape: ', idx_next.shape) if self.verbose else None

            # If stop token is reached, end generation without including the stop token
            if idx_next == (self.vocab_size - 1):
                # print('Ended generation after ', str(i+1), ' iterations.')
                break
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            # print('appended idx_next to original idx, shape: ', idx.shape) if self.verbose else None
            # if i == self.max_sequence_len-1:
            # print('Ended generation after ', str(i+1), ' iterations.')
        return idx

    def forward(self, x, padding_mask=None):
        # torch.manual_seed(21190)
        if self.verbose:
            print("x shape before embedding: ", x.shape)

        # time_before_embed = time.time()
        x = self.embedding_table(x)
        if self.verbose:
            print("x shape after embedding: ", x.shape)
            # print('x after embedding: ', x)
            # print(f'Time for embedding: {time.time()-time_before_embed}')

        time_before_GPTblock = time.time()
        for block in self.GPT_blocks:
            x = block(x, padding_mask=padding_mask)
            if self.verbose:
                print("x after a GPT block: ", x)
        if self.verbose:
            print("x shape after GPT blocks: ", x.shape)
            print(f"Time for GPT blocks: {time.time() - time_before_GPTblock}")
            # print('x after GPT blocks: ', x)

        # return x
        if self.return_embeddings:
            return x

        logits = self.lm_head(x)
        if self.verbose:
            print("Logits shape: ", logits.shape)
            # print(f'Time for LM head: {time.time()-time_before_lmhead}')

        if not self.classify:
            return logits
        else:
            output = self.classification_head(logits)
            return output
