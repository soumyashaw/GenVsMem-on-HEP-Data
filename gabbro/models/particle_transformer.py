"""Wrapper around the official implementation of ParT."""

import os
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.weaver_particle_transformer import ParticleTransformer  # noqa: E402
from gabbro.utils.pylogger import get_pylogger

sys.path.append("/home//home/birkjosc/repositories")

logger = get_pylogger(__name__)

# default kwargs are in configs/model/model_classifier_ParT.yaml
PART_KIN_MODEL_PATH = os.getenv(
    "PART_KIN_MODEL_PATH",
    "/beegfs/desy/user/birkjosc/repositories/particle_transformer/models/ParT_kin.pt",
)
PART_FULL_MODEL_PATH = os.getenv(
    "PART_FULL_MODEL_PATH",
    "/beegfs/desy/user/birkjosc/repositories/particle_transformer/models/ParT_full.pt",
)


class ParTLightning(LightningModule):
    """Lightning module with ParT model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_out_nodes = self.hparams.n_out_nodes

        # has to be named "mod" in order to work with the official ParT implementation ckpt
        self.mod = ParticleTransformer(**model_kwargs)

        # load the weights of an official ParT model if the flag is set
        if self.hparams.get("use_official_ParTkin_ckpt", False) is True:
            logger.info(f"Loading ParT-kin weights from {PART_KIN_MODEL_PATH}")
            if PART_KIN_MODEL_PATH is None:
                raise ValueError(
                    "You are trying to load the ParT-kin weights, but the env variable "
                    "PART_KIN_MODEL_PATH is not set."
                )
            self.load_state_dict(torch.load(PART_KIN_MODEL_PATH, map_location="cuda"))  # nosec

        elif self.hparams.get("use_official_ParTfull_ckpt", False) is True:
            logger.info(f"Loading ParT-full weights from {PART_FULL_MODEL_PATH}")
            if PART_FULL_MODEL_PATH is None:
                raise ValueError(
                    "You are trying to load the ParT-full weights, but the env variable "
                    "PART_FULL_MODEL_PATH is not set."
                )
            self.load_state_dict(torch.load(PART_FULL_MODEL_PATH, map_location="cuda"))  # nosec

        else:
            logger.info("Not loading any pre-trained weights.")

        if self.hparams.reinitialize_final_mlp is True or self.hparams.n_out_nodes != 10:
            logger.info("Reinitializing final MLP layer.")
            self.reinitialize_final_mlp()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

    def reinitialize_final_mlp(self):
        self.mod.fc = torch.nn.Linear(128, self.n_out_nodes)

    def forward(self, features, lorentz_vectors, mask):
        # swap axis 1 and 2 to match the ParT input shape
        features = features.permute(0, 2, 1)
        lorentz_vectors = lorentz_vectors.permute(0, 2, 1)
        mask = mask.unsqueeze(1)
        # print(f"features.shape: {features.shape}")
        # print(f"lorentz_vectors.shape: {lorentz_vectors.shape}")
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")

    def on_train_epoch_start(self) -> None:
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info("`on_train_epoch_start` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} started.")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        X = batch["part_features"][..., :-4]
        X_vectors = batch["part_features"][..., -4:]
        mask = batch["part_mask"]
        jet_labels = batch["jet_type_labels"]
        # one-hot encode the labels
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.n_out_nodes).float()
        logits = self.forward(features=X, lorentz_vectors=X_vectors, mask=mask)
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Noticed that the fine-tuning to Landscape dataset only yields the reported
        # performance when the model is in eval mode during training.
        # --> (check if this is somehow done in weaver as well?)
        if self.hparams.use_eval_mode_in_train:
            self.eval()
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
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
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
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        logger.info("`configure_optimizers` called.")
        if self.hparams.backbone_lr is not None and isinstance(self.hparams.backbone_lr, float):
            backbone_lr = self.hparams.backbone_lr
            logger.info(f">>> Setting backbone_lr to {backbone_lr}.")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.mod.trimmer.parameters(), "lr": backbone_lr},
                    {"params": self.mod.embed.parameters(), "lr": backbone_lr},
                    {"params": self.mod.pair_embed.parameters(), "lr": backbone_lr},
                    {"params": self.mod.blocks.parameters(), "lr": backbone_lr},
                    {"params": self.mod.cls_blocks.parameters(), "lr": backbone_lr},
                    {"params": self.mod.fc.parameters()},
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
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
