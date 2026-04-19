from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.gpt_model import GPT_DecoderStack
from gabbro.models.transformer import (
    ClassAttentionBlock,  # noqa: E402
    NormformerStack,
)
from gabbro.models.weaver_particle_transformer import ParticleTransformer  # noqa: E402
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


class LinearAndAveragePooling(nn.Module):
    """Applies one linear layer and averages over the sequence dimension.

    This is what was used in MPMv1.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_out_nodes = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, mask):
        output = self.fc(x)
        return (output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


class ClassHeadLinAvgPool(nn.Module):
    """Applies the LinearAndAveragePooling head to the input tensor and then
    another linear layer to get the final output (number of classes)."""

    def __init__(self, input_dim, intermediate_dim, output_dim):
        super().__init__()
        self.lin_avg_pool = LinearAndAveragePooling(input_dim, intermediate_dim)
        self.fc = nn.Linear(intermediate_dim, output_dim)
        self.n_out_nodes = output_dim  # needed for some logic in the LightningModule

    def forward(self, x, mask):
        x = self.lin_avg_pool(x, mask)
        return self.fc(x)


class NormformerCrossBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate=0.1, mlp_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.1)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(
            query=class_token, key=x_norm, value=x_norm, key_padding_mask=mask != 1
        )
        return attn_output


class NormformerCrossBlockv2(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_expansion_factor: int = 4,
        num_heads: int = 8,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # TODO: re-implement the unused parameters (currently commented out)
        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=self.dropout_rate
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * mlp_expansion_factor),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.LayerNorm(input_dim * mlp_expansion_factor),
            nn.Linear(input_dim * mlp_expansion_factor, input_dim),
            nn.Dropout(dropout_rate),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        # nn.init.zeros_(self.mlp[-1].weight)
        # nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_pre_attn = self.norm1(x)
        x_post_attn, attn_weights = self.attn(
            query=class_token, key=x_pre_attn, value=x_pre_attn, key_padding_mask=mask != 1
        )
        x_pre_mlp = self.norm2(x_post_attn) + class_token
        x_post_mlp = x_pre_mlp + self.mlp(x_pre_mlp)
        return x_post_mlp


# --------------------------- Particle Flow Network ---------------------------
class ParticleFlow(nn.Module):
    """Definition of the Particle Flow Network."""

    def __init__(
        self,
        input_dim=None,
        n_out_nodes=2,
        n_embed=16,
        n_tokens=None,
        **kwargs,
    ):
        """Initialise Particle Flow Network.

        Parameters
        ----------
        input_dim : int, optional
            Number of features per point.
        n_out_nodes : int, optional
            Number of output nodes.
        n_embed : int, optional
            Number of embedding dimensions, only used if n_tokens is not None.
        n_tokens : int, optional
            Number of codebook entries (i.e. number of different tokens), only
            used if input_dim is None.
        """

        super().__init__()

        if input_dim is None and n_tokens is None:
            raise ValueError("Either input_dim or n_tokens must be specified")

        self.n_out_nodes = n_out_nodes
        self.n_tokens = n_tokens
        self.n_embed = n_embed

        if n_tokens is None:
            self.phi_1 = nn.Linear(input_dim, 100)
        else:
            self.embedding = nn.Embedding(n_tokens, n_embed)
            self.phi_1 = nn.Linear(n_embed, 100)

        self.phi_2 = nn.Linear(100, 100)
        self.phi_3 = nn.Linear(100, 256)
        self.F_1 = nn.Linear(256, 100)
        self.F_2 = nn.Linear(100, 100)
        self.F_3 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, self.n_out_nodes)

    def forward(self, x, mask):
        batch_size, n_points, n_features = x.size()

        # propagate through phi
        if self.n_tokens is not None:
            x = self.embedding(x).squeeze()
        x = F.relu(self.phi_1(x))
        x = F.relu(self.phi_2(x))
        x = F.relu(self.phi_3(x))

        # sum over points dim.
        x_sum = torch.sum(x * mask[..., None], dim=1)

        # propagate through F
        x = F.relu(self.F_1(x_sum))
        x = F.relu(self.F_2(x))
        x = F.relu(self.F_3(x))

        x_out = self.output_layer(x)

        return x_out


class ClassifierPL(LightningModule):
    """Pytorch-lightning module for jet classification."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_class_name: str = "ParticleFlow",
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model_class_name = model_class_name
        if "keep_backbone_fixed" in model_kwargs:
            self.keep_backbone_fixed = model_kwargs["keep_backbone_fixed"]
            model_kwargs.pop("keep_backbone_fixed")
        else:
            self.keep_backbone_fixed = False

        if self.model_class_name == "ParticleFlow":
            self.model = ParticleFlow(**model_kwargs)
        elif self.model_class_name == "ClassifierNormformer":
            self.model = ClassifierNormformer(**model_kwargs)
        elif self.model_class_name == "ParT":
            self.model = ParticleTransformer(**model_kwargs)
        else:
            raise ValueError(f"Model class {model_class_name} not supported.")

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, features, mask):
        return self.model(features, mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        print(f"Epoch {self.trainer.current_epoch} started.")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data containing the input tensor of images and target labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predicted logits.
            - A tensor of target labels.
        """
        X = batch["part_features"]
        mask = batch["part_mask"]
        jet_labels = batch["jet_type_labels"]
        if len(X.size()) == 2:
            X = X.unsqueeze(-1)
        if self.model_class_name == "BackboneWithClasshead":
            X = X.squeeze().long()
        # one-hot encode the labels
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.model.n_out_nodes).float()
        logits = self.forward(X, mask)
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, logits, targets = self.model_step(batch)

        preds = torch.softmax(logits, dim=1)
        self.train_preds_list.append(preds.detach().cpu().numpy())
        self.train_labels_list.append(targets.detach().cpu().numpy())
        self.train_loss_history.append(loss.detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
        )

        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        self.train_preds = np.concatenate(self.train_preds_list)
        self.train_labels = np.concatenate(self.train_labels_list)
        print(f"Epoch {self.trainer.current_epoch} finished.")
        plt.plot(self.train_loss_history)

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.val_preds_list.append(preds.detach().cpu().numpy())
        self.val_labels_list.append(targets.detach().cpu().numpy())
        # update and log metrics
        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
        )
        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")
        self.val_preds = np.concatenate(self.val_preds_list)
        self.val_labels = np.concatenate(self.val_labels_list)

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
        )
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        logger.info("`on_test_epoch_end` called.")
        self.test_preds = np.concatenate(self.test_loop_preds_list)
        self.test_labels = np.concatenate(self.test_loop_labels_list)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        if self.keep_backbone_fixed:
            print("--- Keeping backbone fixed. ---")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.model.module.parameters(), "lr": 0.0},
                    {"params": self.model.classification_head_linear_embed.parameters()},
                    {"params": self.model.classification_head_linear_class.parameters()},
                ]
            )
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class ClassifierNormformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        num_enc_blocks: int = 2,
        num_class_blocks: int = 3,
        fc_params: list = None,
        n_out_nodes: int = None,
        dropout_rate: float = 0.1,
        self_attention_model_class: str = "GPT_Decoder",
        cross_attention_model_class: str = "NormformerCrossBlock",
        identity_init: bool = False,
        **kwargs,
    ):
        super().__init__()

        if n_out_nodes is None:
            self.n_out_nodes = 10
        else:
            self.n_out_nodes = n_out_nodes

        self.fc_params = fc_params
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_enc_blocks = num_enc_blocks
        self.num_class_blocks = num_class_blocks
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.identity_init = identity_init

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        if self.num_enc_blocks == 0:
            self.self_attention_blocks = None
        else:
            if self_attention_model_class == "GPT_Decoder":
                self.self_attention_blocks = GPT_DecoderStack(
                    n_GPT_blocks=self.num_enc_blocks,
                    embedding_dim=self.hidden_dim,
                    attention_dropout=self.dropout_rate,
                    n_heads=self.num_heads,
                    max_sequence_len=128,
                    mlp_dropout=self.dropout_rate,
                    apply_causal_mask=False,
                )
            elif self_attention_model_class == "Normformer":
                self.self_attention_blocks = NormformerStack(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    num_blocks=self.num_enc_blocks,
                    dropout_rate=self.dropout_rate,
                    init_identity=self.identity_init,
                )
            else:
                raise ValueError(
                    f"Self-attention model class {self_attention_model_class} not supported."
                    "Supported choices are 'Normformer' and 'GPT_Decoder'."
                )

        if cross_attention_model_class == "ClassAttentionBlock":
            class_attn_kwargs = dict(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_cfg=dict(
                    expansion_factor=4,
                    dropout_rate=0,
                    norm_before=True,
                    norm_between=True,
                    activation="SiLU",
                ),
                dropout_rate=0,
            )
        else:
            class_attn_kwargs = dict(
                input_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                mlp_expansion_factor=4,
            )

        self.class_attention_blocks = nn.ModuleList(
            [
                eval(cross_attention_model_class)(**class_attn_kwargs)  # nosec
                for _ in range(self.num_class_blocks)
            ]
        )
        self.initialize_classification_head()

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # encode
        x = self.input_projection(x)
        if self.self_attention_blocks is not None:
            x = self.self_attention_blocks(x, mask)
        # concatenate class token and x
        class_token = self.class_token.expand(x.size(0), -1, -1)
        mask_with_token = torch.cat([torch.ones(x.size(0), 1).to(x.device), mask], dim=1)

        # pass through class attention blocks, always use the updated class token
        for block in self.class_attention_blocks:
            x_class_token_and_x_encoded = torch.cat([class_token, x], dim=1)
            class_token = block(
                x=x_class_token_and_x_encoded, class_token=class_token, mask=mask_with_token
            )

        return self.final_mlp(class_token.squeeze(1))

    def initialize_classification_head(self):
        if self.fc_params is None:
            self.final_mlp = nn.Linear(self.hidden_dim, self.n_out_nodes)
        else:
            fc_params = [[self.hidden_dim, 0]] + self.fc_params
            layers = []

            for i in range(1, len(fc_params)):
                in_dim = fc_params[i - 1][0]
                out_dim = fc_params[i][0]
                dropout_rate = fc_params[i][1]
                layers.extend(
                    [
                        nn.Linear(in_dim, out_dim),
                        nn.Dropout(dropout_rate),
                        nn.ReLU(),
                    ]
                )
            # add final layer
            layers.extend([nn.Linear(fc_params[-1][0], self.n_out_nodes)])
            self.final_mlp = nn.Sequential(*layers)
