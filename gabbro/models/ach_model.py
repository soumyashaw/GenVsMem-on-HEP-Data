"""Backbone model with different heads."""

from typing import Any, Dict, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import vector

from gabbro.metrics.utils import calc_accuracy
from gabbro.utils.pylogger import get_pylogger

from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    TransformerEncoderLayer,
    CrossEntropyLoss,
    LayerNorm,
    Dropout,
)

vector.register_awkward()
logger = get_pylogger(__name__)

# fmt: off

class EmbeddingProductHead(Module):
    def __init__(self, hidden_dim=256, num_features=3, num_bins=(41, 41, 41)):
        super(EmbeddingProductHead, self).__init__()
        assert num_features == 3
        self.num_features = num_features
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.combined_bins = int(np.sum(num_bins))
        self.linear = Linear(hidden_dim, self.combined_bins * hidden_dim)
        self.act = torch.nn.Softplus()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        batch_size, seq_len, _ = emb.shape
        bin_emb = self.act(self.linear(emb))
        bin_emb = bin_emb.view(batch_size, seq_len, self.combined_bins, self.hidden_dim)
        bin_emb_x, bin_emb_y, bin_emb_z = torch.split(bin_emb, self.num_bins, dim=2)

        bin_emb_xy = bin_emb_x.unsqueeze(2) * bin_emb_y.unsqueeze(3)
        bin_emb_xy = bin_emb_xy.view(batch_size, seq_len, -1, self.hidden_dim)

        logits = bin_emb_xy @ bin_emb_z.transpose(2, 3)
        logits = self.logit_scale.exp() * logits.view(batch_size, seq_len, -1)
        return logits


class JetTransformerClassifier(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        n_out_nodes=10,
        max_seq_len=100,
        **kwargs,
    ):
        super(JetTransformerClassifier, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.n_out_nodes = n_out_nodes
        self.max_seq_len = max_seq_len

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        self.flat = torch.nn.Flatten()
        self.out = Linear(hidden_dim * max_seq_len, n_out_nodes)
        # self.criterion = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)
        emb = self.flat(emb)
        out = self.out(emb)
        return out

    def loss(self, logits, true_bin):
        loss = self.criterion(logits, true_bin)
        return loss


class JetTransformer(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        output="linear",
        classifier=False,
        tanh=False,
        end_token=False,
    ):
        super(JetTransformer, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.total_bins = int(np.prod(num_bins))
        if end_token:
            self.total_bins += 1
        self.classifier = classifier
        self.tanh = tanh
        print(f"Bins: {self.total_bins}")

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        if output == "linear":
            self.out_proj = Linear(hidden_dim, self.total_bins)
        else:
            self.out_proj = EmbeddingProductHead(hidden_dim, num_features, num_bins)
        self.criterion = CrossEntropyLoss()

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)

        # project final embedding to logits (not normalized with softmax)
        logits = self.out_proj(emb)
        if self.tanh:
            return 13 * torch.tanh(0.1 * logits)
        else:
            return logits

    def loss(self, logits, true_bin):
        # ignore final logits
        logits = logits[:, :-1].reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        loss = self.criterion(logits, true_bin)
        return loss

    def probability(
        self,
        logits,
        padding_mask,
        true_bin,
        perplexity=False,
        logarithmic=False,
        topk=False,
    ):
        batch_size, padded_seq_len, num_bin = logits.shape
        seq_len = padding_mask.long().sum(dim=1)

        # ignore final logits
        logits = logits[:, :-1]
        probs = torch.softmax(logits, dim=-1)

        if topk:
            vals, idx = torch.topk(probs, topk, dim=-1, sorted=False)
            probs = torch.zeros_like(probs, device=probs.device)
            probs[
                torch.arange(probs.shape[0])[:, None, None],
                torch.arange(probs.shape[1])[None, :, None],
                idx,
            ] = vals

            probs = probs / probs.sum(dim=-1, keepdim=True)

        probs = probs.reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        # select probs of true bins
        sel_idx = torch.arange(probs.shape[0], dtype=torch.long, device=probs.device)
        probs = probs[sel_idx, true_bin].view(batch_size, padded_seq_len - 1)
        probs[~padding_mask[:, 1:]] = 1.0
        if perplexity:
            probs = probs ** (1 / seq_len.float().view(-1, 1))

        if logarithmic:
            probs = torch.log(probs).sum(dim=1)
        else:
            probs = probs.prod(dim=1)
        return probs

    def sample(self, starts, device, len_seq, trunc=None):
        def select_idx():
            # Select bin at random according to probabilities
            rand = torch.rand((len(jets), 1), device=device)
            preds_cum = torch.cumsum(preds, -1)
            preds_cum[:, -1] += 0.01  # If rand = 1, sort it to the last bin
            idx = torch.searchsorted(preds_cum, rand).squeeze(1)
            return idx

        if not trunc is None and trunc >= 1:
            trunc = torch.tensor(trunc, dtype=torch.long)

        jets = -torch.ones((len(starts), len_seq, 3), dtype=torch.long, device=device)
        true_bins = torch.zeros((len(starts), len_seq), dtype=torch.long, device=device)

        # Set start bins and constituents
        num_prior_bins = torch.cumprod(torch.tensor([1, 41, 31]), -1).to(device)
        bins = (starts * num_prior_bins.reshape(1, 1, 3)).sum(axis=2)
        true_bins[:, 0] = bins
        jets[:, 0] = starts
        padding_mask = jets[:, :, 0] != -1

        self.eval()
        finished = torch.ones(len(starts)) != 1
        with torch.no_grad():
            for particle in range(len_seq - 1):
                if all(finished):
                    break
                # Get probabilities for the next particles
                preds = self.forward(jets, padding_mask)[:, particle]
                preds = torch.nn.functional.softmax(preds[:, :], dim=-1)

                # Remove low probs
                if not trunc is None:
                    if trunc < 1:
                        preds = torch.where(
                            preds < trunc, torch.zeros(1, device=device), preds
                        )
                    else:
                        preds, indices = torch.topk(preds, trunc, -1, sorted=False)

                preds = preds / torch.sum(preds, -1, keepdim=True)

                idx = select_idx()
                if not trunc is None and trunc >= 1:
                    idx = indices[torch.arange(len(indices)), idx]
                finished[idx == 39401] = True

                # Get tuple from found bin and set next particle properties
                true_bins[~finished, particle + 1] = idx[~finished]
                bins = self.idx_to_bins(idx[~finished])
                for ind, tmp_bin in enumerate(bins):
                    jets[~finished, particle + 1, ind] = tmp_bin

                padding_mask[~finished, particle + 1] = True
        return jets, true_bins

    def idx_to_bins(self, x):
        pT = x % 41
        eta = torch.div((x - pT), 41, rounding_mode="trunc") % torch.div(
            1271, 41, rounding_mode="trunc"
        )
        phi = torch.div((x - pT - 41 * eta), 1271, rounding_mode="trunc")
        return pT, eta, phi


class CNNclass(Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 1 x 30 x 30, Output = 32 x 30 x 30
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 32 x 30 x 30, Output = 32 x 15 x 15
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 32 x 15 x 15, Output = 64 x 15 x 15
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 15 x 15, Output = 64 x 7 x 7
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 64 x 7 x 7, Output = 64 x 7 x 7
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 7 x 7, Output = 64 x 3 x 3
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 3 * 3, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)


class ParticleNet(Module):
    pass

# fmt: off

class BackboneClassificationLightning(L.LightningModule):
    """Backbone with classification head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = JetTransformerClassifier(
            **model_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, X, mask):
        logits = self.module(X, mask)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info(f"Epoch {self.trainer.current_epoch} started.")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        X = batch["part_features"].to("cuda")
        mask = batch["part_mask"].to("cuda")
        jet_labels = batch["jet_type_labels"]
        if len(X.size()) == 2:
            X = X.unsqueeze(-1)
        X = X.squeeze().long()
        # one-hot encode the labels
        logits = self.forward(X, mask)
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.module.n_out_nodes).float()
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        preds = torch.softmax(logits, dim=1)
        self.train_preds_list.append(preds.float().detach().cpu().numpy())
        self.train_labels_list.append(targets.float().detach().cpu().numpy())
        self.train_loss_history.append(loss.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        plt.plot(self.train_loss_history)

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.val_preds_list.append(preds.float().detach().cpu().numpy())
        self.val_labels_list.append(targets.float().detach().cpu().numpy())
        # update and log metrics
        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.float().detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_end(self):
        logger.info("`on_test_epoch_end` called.")
        self.test_preds = np.concatenate(self.test_loop_preds_list)
        self.test_labels = np.concatenate(self.test_loop_labels_list)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")
        if self.hparams.model_kwargs.keep_backbone_fixed:
            logger.info("--- Keeping backbone fixed. ---")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.module.parameters(), "lr": 0.0},
                    {"params": self.head.parameters()},
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
