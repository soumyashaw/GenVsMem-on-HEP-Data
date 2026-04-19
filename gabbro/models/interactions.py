"""Interaction features similar to ParT, but without using physics knowledge."""

import torch
import torch.nn as nn


class PairEmbed(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_pointwise_dims: list,
        mlp_pair_dims: list,
        conv_dims: list,
        activation: str = "gelu",
        use_preactivation_pair: bool = True,
        use_residual_point_cat: bool = False,
        use_diff: bool = False,
    ):
        super().__init__()

        self.use_residual_point_cat = use_residual_point_cat
        self.use_diff = use_diff
        self.input_dim = input_dim

        # MLP that goes from input_dim to mlp_dims[-1]
        mlp_pointwise_dims = [self.input_dim] + mlp_pointwise_dims

        mlp_layers = []
        for i in range(len(mlp_pointwise_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_pointwise_dims[i], mlp_pointwise_dims[i + 1]))
            mlp_layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        self.mlp_pointwise = nn.Sequential(*mlp_layers)

        # MLP that calculates the pairwise features
        dim_factor = 3 if use_diff else 2
        if self.use_residual_point_cat:
            mlp_pair_dims = [
                (mlp_pointwise_dims[-1] + self.input_dim) * dim_factor
            ] + mlp_pair_dims
        else:
            mlp_pair_dims = [mlp_pointwise_dims[-1] * dim_factor] + mlp_pair_dims

        mlp_layers = []
        for i in range(len(mlp_pair_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_pair_dims[i], mlp_pair_dims[i + 1]))
            mlp_layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())

        self.mlp_pair = nn.Sequential(*mlp_layers)

        # Convolutional 1d layers
        conv_dims = [mlp_pair_dims[-1]] + conv_dims
        conv_layers = [
            nn.BatchNorm1d(conv_dims[0]),
        ]
        for i in range(len(conv_dims) - 1):
            conv_layers.extend(
                [
                    nn.Conv1d(conv_dims[i], conv_dims[i + 1], 1),
                    nn.BatchNorm1d(conv_dims[i + 1]),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
        if use_preactivation_pair:
            conv_layers = conv_layers[:-1]  # remove last activation
        self.embed = nn.Sequential(*conv_layers)
        self.out_dim = conv_dims[-1]

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        # embed the features using the MLP (pointwise)
        # print(f"x.shape = {x.shape} (at input)")
        x_after_mlp = self.mlp_pointwise(x)
        # concatenate the processed features with the original features
        if self.use_residual_point_cat:
            x = torch.cat([x_after_mlp, x], dim=-1)  # (batch, input_dim + dim, seq_len)
        else:
            x = x_after_mlp
        # print(f"x.shape = {x.shape} (after mlp_pointwise)")
        # repeat x along the sequence length
        x = x.unsqueeze(1).repeat(1, seq_len, 1, 1)  # (batch, seq_len, seq_len, dim)
        # print(f"x.shape = {x.shape} (after repeat)")

        i, j = torch.tril_indices(
            seq_len,
            seq_len,
            # offset=-1 if self.remove_self_pair else 0,
            device=x.device,
        )
        xi = x[:, i, j, :]  # (batch, dim, seq_len*(seq_len+1)/2)
        xj = x[:, j, i, :]  # (batch, dim, seq_len*(seq_len+1)/2)
        # print(f"xi.shape = {xi.shape}")
        # print(f"xj.shape = {xj.shape}")

        # concatenate the pairwise features such that the features are
        # the concatenation of the two features (xi, xj)
        xij = torch.cat([xi, xj], dim=-1)  # (batch, seq_len*(seq_len+1)/2, 2*dim)
        # print(f"xij.shape = {xij.shape}")
        if self.use_diff:
            xij = torch.cat([xij, xi - xj], dim=-1)  # (batch, seq_len*(seq_len+1)/2, 3*dim)

        # embed the pairwise features using the MLP (pairwise)
        elements = self.mlp_pair(xij)
        # print(f"xij.shape = {xij.shape} (after mlp_pair)")

        # apply convolutional layers
        elements = elements.transpose(1, 2)
        elements = self.embed(elements)
        # print(f"elements.shape = {elements.shape} (after conv)")
        elements = elements.transpose(1, 2)

        y = torch.zeros(
            batch_size,
            seq_len,
            seq_len,
            self.out_dim,
            dtype=elements.dtype,
            device=elements.device,
        )
        y[:, i, j, :] = elements
        y[:, j, i, :] = elements

        return y


class PairEmbedDotProduct(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_pointwise_dims: list,
        n_interaction_matrices: int = 4,
        **kwargs,
    ):
        super().__init__()
        # MLP that goes from input_dim to mlp_dims[-1]
        mlp_pointwise_dims = [input_dim] + mlp_pointwise_dims
        # initialize the MLP
        mlp_layers = []
        for i in range(len(mlp_pointwise_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_pointwise_dims[i], mlp_pointwise_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
        self.mlp_pointwise = nn.Sequential(*mlp_layers)
        self.n_interaction_matrices = n_interaction_matrices
        # check that the output dim of the MLP is divisible by number of interaction matrices
        if mlp_pointwise_dims[-1] % n_interaction_matrices != 0:
            raise ValueError(
                "The output dimension of the MLP must be divisible by the number of interaction matrices."
            )

    def forward(self, x, mask=None):
        # x: (batch, input_dim, seq_len)
        batch_size, seq_len, _ = x.size()
        # embed the features using the MLP (pointwise)
        x_after_mlp = self.mlp_pointwise(x)
        # reshape to (batch, seq_len, n_interaction_matrices, -1)
        x_after_mlp = x_after_mlp.view(batch_size, seq_len, self.n_interaction_matrices, -1)
        # compute dot product between all particles (for each interaction dimension)
        dot_products = torch.einsum("bsai,btai->bsta", x_after_mlp, x_after_mlp)
        # the output dimension is then (batch, seq_len, seq_len, n_interaction_matrices)
        return dot_products
