"""This backbone model uses both the backbone embeddings and continuous features as input."""

from typing import Any, Dict, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.classifiers import ClassifierNormformer
from gabbro.models.gpt_model import BackboneModel
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)

# -------------------------------------------------------------------------
# ------------ BACKBONE + Classification head -----------------------------
# -------------------------------------------------------------------------


class BackboneClassificationHybridHeadLightning(L.LightningModule):
    """Backbone with classification head with the classification head using both the backbone
    embeddings and continuous features that bypass the backbone."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        class_head_type: str = "summation",
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # get the number of dimension for continuous features from the model_kwargs
        self.continuous_hidden_dim = model_kwargs.get("continuous_hidden_dim", 0)
        self.continuous_input_dim = model_kwargs.get("continuous_input_dim", 0)
        self.backbone_hidden_dim = model_kwargs["embedding_dim"] - self.continuous_hidden_dim
        self.n_out_nodes = model_kwargs["n_out_nodes"]
        self.use_backbone_embeddings_as_classifier_input = model_kwargs.get(
            "use_backbone_embeddings_as_classifier_input", True
        )
        self.use_continuous_features_as_classifier_input = model_kwargs.get(
            "use_continuous_features_as_classifier_input", True
        )
        # remove from dict
        model_kwargs.pop("continuous_hidden_dim", None)
        model_kwargs.pop("continuous_input_dim", None)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        if (
            self.use_continuous_features_as_classifier_input
            and self.use_backbone_embeddings_as_classifier_input
        ):
            self.continuous_feature_projection_dim = model_kwargs["classifier_hidden_dim"] // 2
            self.backbone_projection_dim = model_kwargs["classifier_hidden_dim"] // 2
        elif self.use_continuous_features_as_classifier_input:
            self.continuous_feature_projection_dim = model_kwargs["classifier_hidden_dim"]
            self.backbone_projection_dim = 0
        elif self.use_backbone_embeddings_as_classifier_input:
            self.continuous_feature_projection_dim = 0
            self.backbone_projection_dim = model_kwargs["classifier_hidden_dim"]
        else:
            raise ValueError(
                "No input for the classifier head. Looks like both continuous and "
                "backbone embeddings are disabled."
            )

        self.continuous_input_projection = nn.Sequential(
            nn.Linear(
                self.continuous_input_dim,
                self.continuous_feature_projection_dim,
            ),
            # nn.ReLU(),
        )
        self.backbone_projection = nn.Sequential(
            nn.Linear(
                model_kwargs["embedding_dim"],
                self.backbone_projection_dim,
            ),
            # nn.ReLU(),
        )

        self.head = ClassifierNormformer(
            input_dim=model_kwargs["classifier_hidden_dim"],
            hidden_dim=model_kwargs["classifier_hidden_dim"],
            model_kwargs={
                "n_out_nodes": model_kwargs["n_out_nodes"],
            },
            num_heads=model_kwargs.get("head_num_attention_heads", 8),
            num_enc_blocks=model_kwargs.get("head_num_attention_blocks", 5),
            num_class_blocks=model_kwargs.get("head_num_cross_attention_blocks", 2),
            dropout_rate=model_kwargs.get("head_cross_attention_dropout_rate", 0.1),
            class_head_kwargs=model_kwargs.get("final_fc_kwargs", None),
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        logger.info(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        logger.info(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)  # nosec
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

    def forward(self, X, mask):
        # X is of shape (batch_size, seq_len, 1 + n_continuous_features)
        # get the token-id embeddings from the backbone
        # token_id_embeddings_from_backbone = self.module(X[:, :, 0].long(), mask)
        # token_id_embeddings = self.backbone_projection(token_id_embeddings_from_backbone)
        # concatenate
        # x = torch.cat([token_id_embeddings, continuous_embeddings], dim=-1)
        if (
            self.use_backbone_embeddings_as_classifier_input
            and self.use_continuous_features_as_classifier_input
        ):
            x = self.module(X[:, :, 0].long(), mask)
            x = self.backbone_projection(x)
            continuous_input = self.continuous_input_projection(X[:, :, 1:])
            x = torch.cat([x, continuous_input], dim=-1)
        elif self.use_backbone_embeddings_as_classifier_input:
            x = self.module(X[:, :, 0].long(), mask)
            x = self.backbone_projection(x)
        elif self.use_continuous_features_as_classifier_input:
            x = self.continuous_input_projection(X[:, :, 1:])
        else:
            raise ValueError(
                "No input for the classifier head. Looks like both continuous and "
                "backbone embeddings are disabled."
            )

        logits = self.head(x, mask)
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
        # if len(X.size()) == 2:
        #     X = X.unsqueeze(-1)
        # X = X.squeeze().long()
        # one-hot encode the labels
        logits = self.forward(X, mask)
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.n_out_nodes).float()
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
                    {"params": self.continuous_input_projection.parameters()},
                    {"params": self.backbone_projection.parameters()},
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
