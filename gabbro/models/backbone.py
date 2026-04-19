"""Backbone model with different heads."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from gabbro.data.loading import safe_load_features_from_ak_array
from gabbro.metrics.utils import calc_acc_from_logits, calc_accuracy
from gabbro.models.backbone_base import (
    BackboneModel,
    BackboneTransformer,
    MPMHead,
    TokenPredictionHead,
)
from gabbro.models.classifiers import ClassHeadLinAvgPool, ClassifierNormformer
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    combine_ak_arrays,
    convert_torch_token_sequence_with_stop_token_to_ak,
    fix_padded_logits,
    np_to_ak,
    p4s_from_ptetaphimass,
    set_fraction_ones_to_zeros,
)
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import compare_two_pp_dicts

vector.register_awkward()

logger = get_pylogger(__name__)


def simple_optim_sched(model: L.LightningModule) -> dict:
    """Configure the optimizers and learning rate scheduler."""
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = {
        "scheduler": model.hparams.scheduler(optimizer=opt),
        "interval": "step",
    }
    return [opt], [scheduler]


def load_backbone_weights(model, ckpt_path, strict=True):
    """Load the backbone model weights.

    Parameters
    ----------
    model : LightningModule
        The lightning model.
    ckpt_path : str
        Path to the checkpoint file.
    strict : bool, optional
        Whether to load the weights strictly. (default is True)
    """
    # if attached to a trainer, save the state dict from before
    # loading to the default state dict
    if model._trainer is not None:
        logger.info("Saving state dict before loading weights")
        path = f"{model.trainer.default_root_dir}/state_dict_before_loading_backbone_weights.ckpt"
        logger.info(f"Saving state dict to {path}")
        torch.save(model.state_dict(), path)

    logger.info(f"Loading backbone weights from {ckpt_path}")
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)  # nosec
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # drop all keys containing ".tril." in the state_dict to ensure backwards compatibility
    # with the old `tril` definition https://github.com/joschkabirk/gabbro/pull/167
    state_dict = {key: value for key, value in state_dict.items() if ".tril" not in key}
    # drop all keys *not* starting with "module." or "backbone."
    # state_dict = {key: value for key, value in state_dict.items() if "module." in key}
    # keep only keys starting with "module" and remove the "module."
    state_dict = {
        ".".join(key.split(".")[1:]): value
        for key, value in state_dict.items()
        if key.startswith("module.") or key.startswith("backbone.")
    }
    # if the current model has keys with ".tril" included, use them!
    state_dict_tril = {
        key: value for key, value in model.backbone.state_dict().items() if "tril" in key
    }
    state_dict.update(state_dict_tril)
    model.backbone.load_state_dict(state_dict, strict=strict)

    if model._trainer is not None:
        logger.info("Saving state dict after loading weights")
        path = f"{model.trainer.default_root_dir}/state_dict_after_loading_backbone_weights.ckpt"
        logger.info(f"Saving state dict to {path}")
        torch.save(model.state_dict(), path)


def get_combined_index_3d(
    idx_0: torch.Tensor,
    idx_1: torch.Tensor,
    idx_2: torch.Tensor,
    N_0: int,
    N_1: int,
    N_2: int,
    indices_have_plus_one: bool = True,
    return_mask_where_stop_token: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to combine three indices into a single index.

    Parameters
    ----------
    idx_0 : torch.Tensor
        The first index.
    idx_1 : torch.Tensor
        The second index.
    idx_2 : torch.Tensor
        The third index.
    N_0 : int
        The number of bins/indices for the first index. If this is e.g. 10, the index
        should be in the range [0, 9].
    N_1 : int
        The number of bins/indices for the second index.
    N_2 : int
        The number of bins/indices for the third index.
    indices_have_plus_one : bool, optional
        Whether the indices have a +1 offset. (default is True)
        If set to True, the indices are assumed to be in the range [1, N] instead
        of [0, N-1] and are shifted by -1.
    return_mask_where_stop_token : bool, optional
        Whether to return a mask where the stop token is. (default is False)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor] or torch.Tensor
        The combined index and a mask where the stop token is (if
        `return_mask_where_stop_token` is True).
    """
    if indices_have_plus_one:
        idx_0 = idx_0 - 1
        idx_1 = idx_1 - 1
        idx_2 = idx_2 - 1

    combined_idx = idx_0 * N_1 * N_2 + idx_1 * N_2 + idx_2

    if return_mask_where_stop_token:
        max_idx = (N_0) * N_1 * N_2 + (N_1) * N_2 + (N_2)
        return combined_idx, torch.where(
            combined_idx == max_idx, torch.tensor(True), torch.tensor(False)
        )
    return combined_idx


def get_indices_from_combined_index_3d(
    combined_index: torch.Tensor,
    N_0: int,
    N_1: int,
    N_2: int,
    add_plus_one_to_indices: bool = True,
) -> torch.Tensor:
    """Helper function to get the individual indices from a combined index. This is the inverse
    operation of `get_combined_index_3d`.

    Parameters
    ----------
    combined_index : torch.Tensor
        The combined index.
    N_0 : int
        The number of bins/indices for the first index.
    N_1 : int
        The number of bins/indices for the second index.
    N_2 : int
        The number of bins/indices for the third index.
    add_plus_one_to_indices : bool, optional
        Whether to add +1 to the indices. (default is True)

    Returns
    -------
    torch.Tensor
        The individual indices.
    """
    idx_0 = combined_index // (N_1 * N_2)
    idx_1 = (combined_index % (N_1 * N_2)) // N_2
    idx_2 = combined_index % N_2
    if add_plus_one_to_indices:
        return torch.stack([idx_0 + 1, idx_1 + 1, idx_2 + 1], dim=-1)
    return torch.stack([idx_0, idx_1, idx_2], dim=-1)


# -------------------------------------------------------------------------
# ------------ BACKBONE + Generative (next-token-prediction) head ---------
# -------------------------------------------------------------------------


class NextTokenPredictionHead(nn.Module):
    """Head for predicting the next token in a sequence."""

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc1(x)


class BackboneNextTokenPredictionLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        verbose=False,
        alpha_regression_loss: float = 10,
        use_continuous_input: bool = False,
        exclude_padded_values_from_loss: bool = True,
        scheduler_lightning_kwargs: dict = None,
        multi_token_prediction: bool = False,
        backwards_compatibility_mode: bool = False,
        vqvae_weights_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        if "token_dir" in model_kwargs and "token_dir" in kwargs:
            raise ValueError(
                "token_dir is defined in both model_kwargs and kwargs. Should only be specified in `model_kwargs`"
            )

        self.pylogger = get_pylogger(__name__)

        if vqvae_weights_path is not None:
            model_kwargs["token_dir"] = vqvae_weights_path
            self.pylogger.warning(
                "Overriding `token_dir` in `model_kwargs` with `vqvae_weights_path`."
            )

        print(f"Model kwargs: {model_kwargs}")
        if self.hparams.backwards_compatibility_mode:
            self.pylogger.warning(
                "Using backwards compatibility mode. This is not recommended for new models."
                "This allows to load a model that `does not` have the `token_dir` attribute."
            )
            self.token_dir = (
                Path(model_kwargs["token_dir"]) if "token_dir" in model_kwargs else None
            )
        else:
            self.token_dir = Path(model_kwargs["token_dir"])

        # this is just used to simplify the `self.log(...)` calls later on
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.hparams.use_continuous_input:
            self.backbone = BackboneTransformer(**model_kwargs)
        else:
            self.backbone = BackboneModel(**model_kwargs)

        self.vocab_size = model_kwargs["vocab_size"]

        # depending on the number of token groups, the vocab size has to be adjusted
        if not hasattr(self.backbone, "n_token_groups"):
            self.backbone.n_token_groups = 1

        if self.backbone.n_token_groups > 1:
            if len(model_kwargs["n_tokens_list"]) == 0:
                raise ValueError(
                    "n_tokens_list must be specified for multidimensional token prediction"
                )
            elif len(model_kwargs["n_tokens_list"]) != 3:
                # TODO: once this is made more general, adjust the error message
                # (e.g. by using the length of the n_tokens_list and checking if
                # it's the same as the number of token groups)
                raise ValueError(
                    "At the moment, only 3D token prediction is supported in "
                    "multidimensional token prediction"
                )
            vocab_size = np.prod(model_kwargs["n_tokens_list"]) + 2
            if self.vocab_size != vocab_size:
                self.pylogger.warning(
                    f"Vocab size is {self.vocab_size}, but should be {vocab_size} for "
                    f"multidimensional token prediction. Changing vocab size to {vocab_size}."
                )
                self.vocab_size = vocab_size
        else:
            self.vocab_size = model_kwargs["vocab_size"] + model_kwargs.get("num_classes", 0)

        if self.hparams.multi_token_prediction:
            self.head = TokenPredictionHead(
                input_dim=model_kwargs["embedding_dim"],
                vocab_size=self.vocab_size,
                n_pred=self.hparams.multi_token_prediction,
            )
        else:
            self.head = NextTokenPredictionHead(
                embedding_dim=model_kwargs["embedding_dim"],
                vocab_size=self.vocab_size,
            )

        if self.hparams.use_continuous_input:
            logger.info(
                f"Preprocessing dict for continuous features: {self.backbone.particle_features_dict}"
            )
            self.load_vqvae_weights(**model_kwargs)

            logger.info(f"Preprocessing dict for VQ-VAE: {self.vqvae_pp_dict}")
            logger.info("Comparing the preprocessing dicts of the backbone and VQ-VAE model.")
            compare_two_pp_dicts(
                pp_dict_1=self.backbone.particle_features_dict,
                pp_dict_2=self.vqvae_pp_dict,
                ignore_features_not_present_in_second_dict=True,
            )
        # --------------------------------
        # --------------------------------

        if self.hparams.model_kwargs.get("loss_weights", None) is not None:
            weight = torch.ones(self.vocab_size)
            weight[self.vocab_size - 1] = self.hparams.model_kwargs["loss_weights"].get(
                "stop_token", 1.0
            )
            self.criterion = nn.CrossEntropyLoss(weight=weight)
            self.pylogger.info("Using weighted cross-entropy loss")
            self.pylogger.info("Token-ids with non-default weights:")
            for token_id, w in enumerate(weight):
                if w != 1.0:
                    self.pylogger.info(f"- {token_id}: {w}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.verbose = verbose
        self.gen_model_pp_dict = None

        self.train_loss_history = []
        self.val_loss_list = []
        self.val_input_list = []
        self.val_mask_list = []
        self.val_target_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")

        self.pylogger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def forward(self, x, mask=None, x_jet=None, return_logits_only=False):
        backbone_out = self.backbone(x, mask, x_jet=x_jet)
        if self.hparams.multi_token_prediction:
            logits = self.head(backbone_out, mask=mask)
        else:
            logits = self.head(backbone_out)
        if self.verbose:
            self.pylogger.info("Logits shape: ", logits.shape)
        return logits

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        """

        # all token-ids up to the last one are the input, the ones from the second
        # to the (including) last one are the target
        # this model step uses the convention that the first particle feature
        # is the token, with the tokens up to the last one
        # the second particle feature is the target token (i.e. the next token)

        X = batch["part_features"]
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None
        mask = batch["part_mask"]
        part_targets = batch.get("part_labels", None)

        if self.hparams.use_continuous_input:
            if part_targets is not None:
                targets = part_targets.long()
                input = X
                # zero-padd them again cause zero-padding is screwed up by preprocessing
                input[:, 0, :] = 0
            else:
                targets = X[:, :, 0].long()
                input = X
                # zero-padd them again cause zero-padding is screwed up by preprocessing
                input[:, 0, 1:] = 0
        else:
            X = X.squeeze().long()
            # we use the convention that token inputs are given in an alternating
            # manner, i.e. the first feature is token1, the second feature is token1_target,
            # the third feature is token2, the fourth feature is token2_target, and so on
            input = X[:, :, 0::2]
            targets = X[:, :, 1::2]
            if self.backbone.n_token_groups > 1:
                # calculate the combined index for the multidimensional token prediction
                targets, mask_where_stop = get_combined_index_3d(
                    targets[:, :, 0],
                    targets[:, :, 1],
                    targets[:, :, 2],
                    *self.hparams.model_kwargs.n_tokens_list,
                    indices_have_plus_one=True,
                    return_mask_where_stop_token=True,
                )
                # add +1 because the combined indices act as if the tokens are the non-shifted ones
                targets += 1
                targets[mask_where_stop] = self.head.vocab_size - 1  # set the stop token
                # apply the mask to the targets (to remove negative values from `get_combined_index_3d`)
                targets = targets * mask

        # compute the logits (i.e. the predictions for the next token)
        if not self.hparams.multi_token_prediction:
            logits = self.forward(input, mask, x_jet=X_jet)
            if self.hparams.exclude_padded_values_from_loss:
                logits = fix_padded_logits(logits, mask.bool(), factor=1e6)

            # reshape the logits and targets to work with the loss function
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.contiguous().view(B * T)

            # calculate the loss terms
            loss_next_token_prediction = self.criterion(logits_reshaped, targets_reshaped)
            mtp_losses = {}
        else:
            logits_mtp = self.forward(input, mask, x_jet=X_jet)

            # treat next token prediction for backwarts compatibility
            logits = logits_mtp[..., 0]
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.contiguous().view(B * T)

            # if self.hparams.exclude_padded_values_from_loss:
            #     logits_mtp = [
            #         fix_padded_logits(logits[i], mask, factor=1e6) for logits in logits_mtp
            #     ]

            start_indices = list(range(logits_mtp.size(-1)))
            end_indices = list(range(logits_mtp.size(-1)))
            start_indices[0] = end_indices[0] = None
            for i in range(1, len(start_indices)):
                end_indices[i] = int(-i)

            # calculate next token prediction loss (where the different heads predict
            # the i'th next token)

            # actual next token prediction loss
            loss_next_token_prediction = self.criterion(
                logits_mtp[..., 0].reshape(B * T, C),
                targets.contiguous().reshape(B * T),
            )

            # add the other token predictions (what comes with multi-token-prediction)
            mtp_losses = {}
            for i in range(1, len(start_indices)):
                mtp_losses[f"loss_ntp_{i + 1}"] = self.criterion(
                    logits_mtp[:, : end_indices[i], :, i].reshape(B * (T - i), C),
                    targets[:, start_indices[i] :].contiguous().reshape(B * (T - i)),
                )

        if self.hparams.multi_token_prediction:
            loss = (loss_next_token_prediction + sum(mtp_losses.values())) / logits_mtp.size(-1)
        else:
            loss = loss_next_token_prediction

        max_len_in_batch = logits.size(1)
        default_token_indices_to_calc = [0, 1, 2, 3, 10, 20, 30, 40, 50, 100]
        token_indices_to_calc = [
            index for index in default_token_indices_to_calc if index < max_len_in_batch
        ]

        acc_dict = calc_acc_from_logits(
            logits, targets, mask, token_indices_to_calc=token_indices_to_calc
        )

        return {
            "loss": loss,
            "loss_next_token_prediction": loss_next_token_prediction,
            "X": X,
            "logits_reshaped": logits_reshaped,
            "targets_reshaped": targets_reshaped,
            "acc_dict": acc_dict,
            "mtp_losses": mtp_losses,
        }

    @torch.no_grad()
    def generate_batch(self, batch_size):
        """Generate a batch of jet constituents autoregressively.

        Parameters
        ----------
        batch_size : int
            Number of jets to generate.

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (batch_size, <var>).
        """
        # idx is (B, T) array of indices in the current context, initialized with the start token
        # thus idx has shape (B, 1) at the beginning
        device = next(self.backbone.parameters()).device  # get the device of the model

        # initialize the idx tensor with the start token
        if self.backbone.n_token_groups > 1:
            # idx is what we use as autoregressive input, and idx_gen is what we use to store the
            # generated tokens
            # (in the multi-dim token case, this is different, because we generate the combined
            # while still having the individual indices as input)
            idx = torch.zeros(batch_size, 1, self.backbone.n_token_groups).long().to(device)
            idx_gen = torch.zeros(batch_size, 1).long().to(device)
        else:
            idx = (torch.ones(batch_size, 1) * self.start_token).long().to(device)

        for i in range(self.backbone.max_sequence_len):
            # get the predictions for the next token
            if self.hparams.multi_token_prediction:
                logits = self(idx)[..., 0]
            else:
                logits = self(idx)

            if self.verbose:
                self.pylogger.info("Logit shape input for generation: ", logits.shape)

            # only look at next-token prediction of last token
            logits = logits[:, -1, ...]  # (B, T, C) becomes (B, C)
            # apply softmax to get probabilities, and exclude the start-token (index 0)
            # (otherwise it can happen, that the start token is predicted as the next token)
            probs = F.softmax(logits[..., 1:], dim=-1)  # (B, C-1)
            idx_next = torch.multinomial(probs, num_samples=1) + 1  # (B, 1)

            if self.backbone.n_token_groups > 1:
                # TODO: implement a check at __init__ that n_token_groups  corresponds
                # to entries in list get the indices for the token groups
                idx_gen = torch.cat((idx_gen, idx_next), dim=1)  # (B, T+1)
                idx_next = get_indices_from_combined_index_3d(
                    idx_next,
                    *self.hparams.model_kwargs.n_tokens_list,
                    add_plus_one_to_indices=True,
                )

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            self.pylogger.info(
                "appended idx_next to original idx, shape: ", idx.shape
            ) if self.verbose else None

        # TODO: I think for multiple token groups it should also be `idx`?
        # --> check this and adapt the code accordingly if needed
        if self.backbone.n_token_groups > 1:
            gen_batch_np = idx_gen.detach().cpu().numpy()
        else:
            gen_batch_np = idx.detach().cpu().numpy()

        gen_batch_ak = ak.from_numpy(gen_batch_np)
        gen_batch_until_stop = []

        # NOTE: the thing below gets the job done, but could be improved
        # remove everything after the first stop token if it exists in the jet
        # loop over the jets in the batch, and only keep the tokens until the stop token
        for jet in gen_batch_ak:
            stop_token_position = np.where(jet == self.head.vocab_size - 1)
            if len(stop_token_position[0]) > 0:
                stop_token_position = stop_token_position[0][0]
            else:
                stop_token_position = jet.shape[0]
            gen_batch_until_stop.append(jet[:stop_token_position])

        # --- Multi-dim tokens ---
        # if the model predicts multiple token groups, we have to translate the combined
        # token indices into the individual token groups
        if self.backbone.n_token_groups > 1:
            # get the maximum length of the jets in the batch to pad only to the
            # maximum length
            max_len = int(ak.max(ak.num(gen_batch_until_stop)))

            # pad the jets to the maximum length and get mask to apply it later
            gen_batch_until_stop_named = ak.Array({"part_token_id": gen_batch_until_stop})
            gen_batch_until_stop_named_padded, gen_batch_mask = ak_pad(
                gen_batch_until_stop_named,
                maxlen=max_len,
                return_mask=True,
            )
            # convert to torch tensor to use the `get_indices_from_combined_index_3d` function
            gen_batch_until_stop_torch = torch.from_numpy(
                ak_to_np_stack(
                    gen_batch_until_stop_named_padded,
                    names=["part_token_id"],
                )
            )
            # get the individual token groups
            gen_batch_until_stop_np_individual = (
                get_indices_from_combined_index_3d(
                    gen_batch_until_stop_torch - 1,
                    *self.hparams.model_kwargs.n_tokens_list,
                    add_plus_one_to_indices=True,
                )
                .squeeze()
                .numpy()
            )
            # convert back to awkward array
            gen_batch_until_stop = np_to_ak(
                gen_batch_until_stop_np_individual,
                names=[f"part_token_id_group_{i}" for i in range(self.backbone.n_token_groups)],
                mask=gen_batch_mask,
                dtype="int64",
            )
        # ---

        return ak.Array(gen_batch_until_stop)

    # TODO: Adapt to allow multidim tokens
    @torch.no_grad()
    def generate_batch_multiclass(self, batch_size):
        """Generate a batch of jet constituents autoregressively.

        Parameters
        ----------
        batch_size : int
            Number of jets to generate.
        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (batch_size, <var>).
        """
        # idx is (B, T) array of indices in the current context, initialized with the start token
        # thus idx has shape (B, 1) at the beginning
        device = next(self.backbone.parameters()).device  # get the device of the model

        if (
            self.class_token is not None  # This means that we want to generate a specific class
        ):  # Initialize both start token and class token, shape (B, 2)
            idx = (
                torch.cat(
                    (torch.zeros(batch_size, 1), torch.ones(batch_size, 1) * self.class_token),
                    dim=1,
                )
                .long()
                .to(device)
            )
            max_sequence_len = self.backbone.max_sequence_len - 2  # We have already added 2 tokens
            first_run = False  # Needed to handle both the with and without specifying class token cases together below
        else:
            idx = torch.zeros(batch_size, 1).long().to(device)  # Only initialize start token
            max_sequence_len = self.backbone.max_sequence_len - 1  # We have already added 1 token

        if self.n_classes is not None and self.class_token is None:
            # If it has trained with class tokens but we don't specify a jet type to generate
            first_run = True

        for i in range(max_sequence_len):
            # get the predictions for the next token
            logits = self(idx)
            self.pylogger.info(
                "Logit shape input for generation: ", logits.shape
            ) if self.verbose else None
            # only look at next-token prediction of last token
            logits = logits[:, -1, ...]  # (B, T, C) becomes (B, C)
            # apply softmax to get probabilities, and exclude the start-token (index 0)
            # (otherwise it can happen, that the start token is predicted as the next token)
            probs = F.softmax(logits[..., 1:], dim=-1)  # (B, C-1)

            # sample from the distribution
            if (self.n_classes is not None) and (self.class_token is None) and first_run:
                # The model has trained with class tokens, but we don't want to specify a class when generating.
                # It then needs to generate its own start token, for the first iteration.
                # Note: we don't force it to choose a class token rather than some other token, but if it is
                # properly trained it will choose to generate class tokens at this step.
                idx_next = torch.multinomial(probs, num_samples=1) + 1  # (B, 1)
                first_run = False
                # For the next iteration, it will use the generation statement below, which is also
                # used for the case where we have provided it with a specific class token.
            elif (self.n_classes is not None) and (not first_run):
                # The model has trained with class tokens, and idx has obtained a start token and a class
                # token (either by user choice or its own generation).
                # Exclude the class tokens and shift the tokens back so they get the correct token value.
                idx_next = (
                    torch.multinomial(probs[:, (self.n_classes) :], num_samples=1)
                    + 1
                    + self.n_classes
                )
            else:
                # Shift the tokens so they get the correct token value
                idx_next = torch.multinomial(probs, num_samples=1) + 1  # (B, 1)

            # TODO: This needs to be fixed for the multidim case
            # # append sampled index to the running sequence
            # if self.module.n_token_groups > 1:
            #     # TODO: implement a check at __init__ that n_token_groups corresponds to entries in list
            #     # get the indices for the token groups
            #     idx_gen = torch.cat((idx_gen, idx_next), dim=1)  # (B, T+1)
            #     idx_next = get_indices_from_combined_index_3d(
            #         idx_next,
            #         *self.hparams.model_kwargs.n_tokens_list,
            #         add_plus_one_to_indices=True,
            #     )
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            self.pylogger.info(
                "appended idx_next to original idx, shape: ", idx.shape
            ) if self.verbose else None

        # TODO: the thing below gets the job done, but could be improved
        # remove everything after the first stop token if it exists in the jet
        # TODO: fix for the multidim case
        # if self.module.n_token_groups > 1:
        #     gen_batch_np = idx_gen.detach().cpu().numpy()
        # else:
        gen_batch_np = idx.detach().cpu().numpy()

        gen_batch_ak = ak.from_numpy(gen_batch_np)
        gen_batch_until_stop = []

        # loop over the jets in the batch, and only keep the tokens until the stop token
        for jet in gen_batch_ak:
            stop_token_position = np.where(jet == self.head.vocab_size - 1)
            if len(stop_token_position[0]) > 0:
                stop_token_position = stop_token_position[0][0]
            else:
                stop_token_position = jet.shape[0]
            gen_batch_until_stop.append(jet[:stop_token_position])

        # if the model predicts multiple token groups, we have to translate the combined
        # token indices into the individual token groups
        if self.backbone.n_token_groups > 1:
            # convert to torch tensor for conversion
            # get the maximum length of the jets in the batch to pad only to the
            # maximum length
            max_len = int(ak.max(ak.num(gen_batch_until_stop)))

            # pad the jets to the maximum length and get mask to apply it later
            gen_batch_until_stop_named = ak.Array({"part_token_id": gen_batch_until_stop})
            gen_batch_until_stop_named_padded, gen_batch_mask = ak_pad(
                gen_batch_until_stop_named,
                maxlen=max_len,
                return_mask=True,
            )
            # convert to torch tensor to use the `get_indices_from_combined_index_3d` function
            gen_batch_until_stop_torch = torch.from_numpy(
                ak_to_np_stack(
                    gen_batch_until_stop_named_padded,
                    names=["part_token_id"],
                )
            )
            # get the individual token groups
            gen_batch_until_stop_np_individual = (
                get_indices_from_combined_index_3d(
                    gen_batch_until_stop_torch - 1,
                    *self.hparams.model_kwargs.n_tokens_list,
                    add_plus_one_to_indices=True,
                )
                .squeeze()
                .numpy()
            )
            # convert back to awkward array
            gen_batch_until_stop = np_to_ak(
                gen_batch_until_stop_np_individual,
                names=[f"part_token_id_group_{i}" for i in range(self.backbone.n_token_groups)],
                mask=gen_batch_mask,
                dtype="int64",
            )

        return ak.Array(gen_batch_until_stop)

    def generate_n_jets_batched(
        self,
        n_jets,
        batch_size,
        class_token=None,
        n_classes=None,
        saveas=None,
        seed=None,
        x_jet=None,
        start_token=0,
    ):
        """Generate jets in batches.

        Parameters
        ----------
        n_jets : int
            Number of jets to generate.
        batch_size : int
            Batch size to use during generation (use as large as possible with memory.)
        class_token: int, optional
            If you want to generate a specific class, enter its token here.
        n_classes: int, optional
            If model has trained with class_token_dict, enter how many classes
            there are in total in the trained model.
        start_token: int, optional
            If a custom start token should be used instead of 0.
        saveas : str, optional
            Path to save the generated jets to (in parquet format). (default is None)
        x_jet : torch.Tensor, optional
            The jet features to use as input. (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (n_jets, <var>).
        """

        self.pylogger.info(f"Generating {n_jets} jets in batches.")

        if x_jet is not None:
            self.pylogger.info("Using x_jet as input for generation.")
            self.pylogger.info(f"x_jet shape: {x_jet.shape}")
            if x_jet.size(0) < n_jets:
                self.pylogger.warning(
                    f"x_jet has fewer jets than specified n_jets={n_jets}. "
                    f"Will generate the only {x_jet.size(0)} jets."
                )
                n_jets = x_jet.size(0)

            if n_jets < batch_size:
                batch_size = n_jets
                self.pylogger.warning(
                    f"Batch size is larger than n_jets. Setting batch size to {batch_size}."
                )
            n_batches = n_jets // batch_size

        else:
            n_batches = (
                n_jets // batch_size + 1 if n_jets % batch_size != 0 else n_jets // batch_size
            )

        generated_jets = []

        self.start_token = start_token
        if seed is not None:
            L.seed_everything(seed)

        self.class_token = class_token
        self.n_classes = n_classes

        self.pylogger.info(
            f"Generating {n_batches * batch_size} jets in {n_batches} batches of size {batch_size}, starting from start token {self.start_token}"
        )

        for i in tqdm(range(n_batches)):
            if self.n_classes is not None:
                gen_batch_ak = self.generate_batch_multiclass(batch_size)
            else:
                if x_jet is not None:
                    x_jet_batch = x_jet[batch_size * i : batch_size * (i + 1)]
                else:
                    x_jet_batch = None
                gen_batch_ak = (
                    self.generate_batch(batch_size)
                    if not self.hparams.use_continuous_input
                    else self.generate_batch_continuous(batch_size, x_jet=x_jet_batch)
                )
            generated_jets.append(gen_batch_ak)

        # concatenate the generated batches
        generated_jets = ak.concatenate(generated_jets)[:n_jets]

        if saveas is not None:
            self.pylogger.info(f"Saving generated jets to {saveas}")
            ak.to_parquet(generated_jets, saveas)

        return generated_jets

    @torch.no_grad()
    def generate_batch_continuous(
        self,
        batch_size,
        return_more=False,
        verbose=False,
        x_jet=None,
    ):
        """Generate a batch of jet constituents autoregressively.

        Parameters
        ----------
        batch_size : int
            Number of jets to generate.
        return_more : bool, optional
            Whether to return more information than just the generated jets. (default is False)
        verbose : bool, optional
            Whether to print more information during generation. (default is False)
        x_jet : torch.Tensor, optional
            The jet features to use as input. (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (batch_size, <var>).
        """
        if x_jet is not None:
            assert x_jet.size(0) == batch_size, "x_jet must have the same batch size as batch_size"

        if hasattr(self.backbone, "continuous_input_dim"):
            continuous_input_dim = self.backbone.continuous_input_dim
        elif hasattr(self.backbone, "part_features_input_dim"):
            continuous_input_dim = len(self.backbone.particle_features_dict)
        else:
            raise ValueError(
                "The model does not have a continuous input dimension. "
                "Please check the model definition."
            )

        if verbose:
            self.pylogger.info(f"Generating a batch of {batch_size} jets.")

        # idx is (B, T) array of indices in the current context, initialized with the start token
        # thus idx has shape (B, 1) at the beginning
        device = next(self.backbone.parameters()).device  # get the device of the model
        # we start with a zero-padded tensor of the continuous features
        continuous_input = torch.zeros(batch_size, 1, continuous_input_dim).to(device)

        # TODO: why not use self.module.max_sequence_len here as done in the non-continuous case?
        for i in range(self.max_sequence_len):
            # get the predictions for the next token
            if verbose:
                self.pylogger.info(f"Continuous input shape: {continuous_input.shape}")
            logits = self.forward(
                continuous_input,
                return_logits_only=True,
                x_jet=x_jet,
            )
            if self.hparams.multi_token_prediction:
                logits = logits[..., 0]

            if verbose:
                self.pylogger.info(f"Logit shape input for generation: {logits.shape}")

            # only look at next-token prediction of last token
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)
            # TODO: implement temperature scaling here (which then also allows to
            # make generation deterministic for debugging)
            # logits = logits / temperature

            # we want to exclude token-id 0 in the prediction, because
            # we'll shift them by -1 later on to get the VQ-VAE tokens, and
            # if we predict 0, we'll get an invalid "-1" token
            probs = F.softmax(logits[:, 1:], dim=-1)  # (B, C-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx_next = idx_next + 1  # shift the indices by +1 to get the correct token
            # append sampled index to the running sequence
            if i == 0:
                idx = idx_next
            else:
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if verbose:
                self.pylogger.info(f"max value in predicted idx: {torch.max(idx)}")

            # convert the token-ids to an awkward array
            gen_batch_tokens_ak = convert_torch_token_sequence_with_stop_token_to_ak(
                idx,
                stop_value=(self.vocab_size - 1),
            )
            if self.verbose:
                self.pylogger.info(
                    "Min and max length of gen_batch_tokens_ak: "
                    f"{ak.min(ak.num(gen_batch_tokens_ak))}, {ak.max(ak.num(gen_batch_tokens_ak))}"
                )
                self.pylogger.info(f"Unique generated idx: {torch.unique(idx).size()[0]}")

            # reconstruct the continuous features from the token-ids
            gen_batch_continuous_ak = self.vqvae_model.reconstruct_ak_tokens(
                gen_batch_tokens_ak - 1,  # -1 because the VQ-VAE tokens are shifted by -1
                pp_dict=self.vqvae_pp_dict,
                hide_pbar=True,
            )
            # add zero-padded features if using features that are not part of the VQ-VAE
            gen_batch_continuous_ak = safe_load_features_from_ak_array(
                ak_array=gen_batch_continuous_ak,
                features=self.backbone.particle_features_dict.keys(),
                load_zeros_if_not_present=True,
            )

            if self.hparams.model_kwargs.interaction_cfg is not None:
                # in case interactions are used, we potentially need the
                # p4s of the particles (which are not used as node-input then)
                # line below assumes `part_pt`, `part_etarel`, `part_phirel`
                # and optionally `part_mass`
                # --> should be made more general
                p4s_vector = p4s_from_ptetaphimass(gen_batch_continuous_ak)
                p4s_ak = ak.Array(
                    {
                        "part_px": p4s_vector.px,
                        "part_py": p4s_vector.py,
                        "part_pz": p4s_vector.pz,
                        "part_energy": p4s_vector.E,
                    }
                )
                gen_batch_continuous_ak = combine_ak_arrays(
                    gen_batch_continuous_ak,
                    p4s_ak,
                )
            # apply the preprocessing before feeding it into the model
            pp_dict = self.backbone.particle_features_dict
            # add p4s to the gen_batch_continuous_ak
            gen_batch_continuous_ak_preprocessed = ak_select_and_preprocess(
                ak_array=gen_batch_continuous_ak,
                pp_dict=pp_dict,
            )
            if verbose:
                self.pylogger.info(gen_batch_continuous_ak_preprocessed)

            # convert to torch tensor to put into the model
            # pad the continuous features to the maximum sequence length
            gen_batch_continuous_ak_padded, gen_batch_mask = ak_pad(
                gen_batch_continuous_ak_preprocessed,
                maxlen=i + 1,
                return_mask=True,
            )
            continuous_input_new = torch.from_numpy(
                ak_to_np_stack(
                    gen_batch_continuous_ak_padded,
                    names=gen_batch_continuous_ak_preprocessed.fields,
                )
            ).to(self.device)

            # add the latest cotinuous column to the continuous_input tensor
            continuous_input = torch.cat(
                [continuous_input, continuous_input_new[:, -1:, :]], dim=1
            )

        # combine the token-ids and the continuous features
        gen_batch_particles_ak = combine_ak_arrays(
            ak.Array({"part_token_id": gen_batch_tokens_ak}),
            gen_batch_continuous_ak,
        )

        if return_more:
            return (
                gen_batch_particles_ak,
                gen_batch_mask,
                idx,
            )

        return gen_batch_particles_ak

    def convert_valtest_batches_to_ak(self, stage):
        """Convert the collected validation loop batches to awkward arrays.

        Returns
        -------
        ak.Array
            The validation input as an awkward array.
        """
        if stage == "val":
            input = np.concatenate(self.val_input_list)
            target = (
                np.concatenate(self.val_target_list) if len(self.val_target_list) > 0 else None
            )
            mask = np.concatenate(self.val_mask_list)
        elif stage == "test":
            input = np.concatenate(self.test_input_list)
            target = (
                np.concatenate(self.test_target_list) if len(self.test_target_list) > 0 else None
            )
            mask = np.concatenate(self.test_mask_list)
        else:
            raise ValueError(f"Stage {stage} not recognized.")

        ak_arr_input = np_to_ak(
            # exclude the start token / dummy continuous feature at the beginning
            np.concatenate(
                [
                    # remove the last token, which in this array is not the stop token
                    # (but instead just the last padded token), but since we are
                    # cropping the mask (a few lines below) at the beginning
                    # this effectively removes the stop token
                    target[:, :-1],
                    input[:, 1:],  # exclude zero-padded dummy continuous feature
                ],
                axis=-1,
            ),
            names=["part_token_id"] + list(self.backbone.particle_features_dict.keys()),
            mask=mask[:, 1:],
        )
        # invert the preprocessing
        pp_dict = {"part_token_id": {}} | self.backbone.particle_features_dict
        ak_arr_input = ak_select_and_preprocess(ak_arr_input, pp_dict, inverse=True)
        return ak_arr_input

    def load_vqvae_weights(self, **model_kwargs):
        """Load the VQ-VAE model weights.

        This is only used for the continuous input case, because there we reconstruct
        tokens "on the fly" in the generation process.

        Parameters
        ----------
        **model_kwargs : dict

        """

        # add the vqvae model to the lightning module but detach its parameters
        # the checkpoint is saved in the main directory of the data directory
        vqvae_ckpt = self.token_dir / "model_ckpt.ckpt"
        self.pylogger.info(f"Loading VQ-VAE model from {vqvae_ckpt}")
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt)
        self.max_sequence_len = model_kwargs["max_sequence_len"]

        self.pylogger.info(
            "Detaching the parameters of the VQ-VAE model from the gradient computation."
        )
        for param in self.vqvae_model.parameters():
            param.requires_grad = False

        # get the preprocessing dict used in the VQ-VAE model training
        vqvae_config_file = self.token_dir / "config.yaml"
        cfg = OmegaConf.load(vqvae_config_file)
        self.vqvae_pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict"])
        if "feature_dict_jet" in cfg.data.dataset_kwargs_common:
            if cfg.data.dataset_kwargs_common["feature_dict_jet"] is not None:
                self.vqvae_pp_dict_jet = OmegaConf.to_container(
                    cfg.data.dataset_kwargs_common["feature_dict_jet"]
                )
            else:
                self.vqvae_pp_dict_jet = None
        # the convention is that tokenized jets have vq-vae token + 1, which means
        # that 0 is the start token and num_codes + 1 is the stop token
        self.stop_token_value = self.vqvae_model.model.vq_kwargs["num_codes"] + 1

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        if self.hparams.get("generate_jets_only", False):
            model_step_output = self.model_step(batch)
            loss = model_step_output["loss"]
            self.log("test_loss", loss.item(), **self.log_dict)
        self._collect_batch_data(batch, "test")

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)

        loss = model_step_output["loss"]
        loss_ntp = model_step_output["loss_next_token_prediction"]
        acc_dict = model_step_output["acc_dict"]
        mtp_losses = model_step_output.get("mtp_losses", {})

        self.log(f"{stage}_loss", loss.item(), **self.log_dict)
        self.log(f"{stage}_loss_ntp", loss_ntp.item(), **self.log_dict)

        for key, value in acc_dict.items():
            self.log(f"{stage}_" + key, value, **self.log_dict)
        for key, value in mtp_losses.items():
            self.log(f"{stage}_" + key, value.item(), **self.log_dict)

        if stage == "train":
            self.train_loss_history.append(float(loss))
        else:
            self._collect_batch_data(batch, stage)

        return loss

    def _collect_batch_data(self, batch, stage: str) -> None:
        """Collect batch data for validation or test stages."""
        if stage == "val":
            self.val_input_list.append(batch["part_features"].float().detach().cpu().numpy())
            self.val_mask_list.append(batch["part_mask"].float().detach().cpu().numpy())
            if batch.get("part_labels") is not None:
                self.val_target_list.append(batch["part_labels"].float().detach().cpu().numpy())
            if hasattr(self.backbone, "jet_features_input_dim"):
                if self.backbone.jet_features_input_dim > 0:
                    self.val_input_list_jet.append(batch["jet_features"].detach().cpu().numpy())
        elif stage == "test":
            self.test_input_list.append(batch["part_features"].float().detach().cpu().numpy())
            self.test_mask_list.append(batch["part_mask"].float().detach().cpu().numpy())
            if batch.get("part_labels") is not None:
                self.test_target_list.append(batch["part_labels"].float().detach().cpu().numpy())
            if hasattr(self.backbone, "jet_features_input_dim"):
                if self.backbone.jet_features_input_dim > 0:
                    self.test_input_list_jet.append(batch["jet_features"].detach().cpu().numpy())

    def on_train_start(self) -> None:
        self._log_start("train")
        if self.backbone_weights_path and self.backbone_weights_path != "None":
            load_backbone_weights(self, self.backbone_weights_path, strict=True)
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        self._log_epoch_start("train")
        self.epoch_train_start_time = time.time()

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                self.pylogger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Current step: {self.global_step}. Current loss: {self.train_loss_history[-1]}. "
                    f"Rank: {self.global_rank}"
                )

    def on_validation_epoch_start(self) -> None:
        self._log_epoch_start("val")
        self.val_input_list, self.val_mask_list, self.val_input_list_jet, self.val_target_list = (
            [],
            [],
            [],
            [],
        )

    def on_test_epoch_start(self) -> None:
        self._log_epoch_start("test")
        (
            self.test_input_list,
            self.test_mask_list,
            self.test_input_list_jet,
            self.test_target_list,
        ) = [], [], [], []

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_end("val")

    def on_test_epoch_end(self):
        self._log_epoch_end("test")

    def _log_start(self, stage: str) -> None:
        self.pylogger.info(f"`on_{stage}_start` called.")
        if stage == "train":
            self.pylogger.info("Setting up the logger with the correct rank.")
            self.pylogger = get_pylogger(__name__, rank=self.trainer.global_rank)
            self.pylogger.info("Logger set up.")

    def _log_epoch_start(self, stage: str) -> None:
        self.pylogger.info(f"`on_{stage}_epoch_start` called.")
        if stage == "train":
            self.pylogger.info(f"Epoch {self.trainer.current_epoch} starting.")

    def _log_epoch_end(self, stage: str) -> None:
        self.pylogger.info(f"`on_{stage}_epoch_end` called.")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        self.pylogger.info("`configure_optimizers` called.")

        if self.hparams.model_kwargs.get("keep_embedding_table_fixed", False):
            logger.info("--- Keeping embedding table fixed. ---")
            backbone_parameters_embedding_table = [
                params
                for name, params in self.backbone.named_parameters()
                if "embedding_table" in name
            ]
            backbone_parameters_other = [
                params
                for name, params in self.backbone.named_parameters()
                if "embedding_table" not in name
            ]
            optimizer = self.hparams.optimizer(
                [
                    {"params": backbone_parameters_embedding_table, "lr": 0.0},
                    {"params": backbone_parameters_other},
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
                    **self.hparams.scheduler_lightning_kwargs,
                },
            }

        return {"optimizer": optimizer}


# -------------------------------------------------------------------------
# ------------------ BACKBONE + Classification head -----------------------
# -------------------------------------------------------------------------


class ClassificationHead(torch.nn.Module):
    """Classification head for the backbone model."""

    def __init__(self, model_kwargs={"n_out_nodes": 2}):
        super().__init__()
        self.backbone_weights_path = None

        if "n_out_nodes" not in model_kwargs:
            model_kwargs["n_out_nodes"] = 2
        if "return_embeddings" not in model_kwargs:
            model_kwargs["return_embeddings"] = True

        self.n_out_nodes = model_kwargs["n_out_nodes"]
        model_kwargs.pop("n_out_nodes")

        self.classification_head_linear_embed = nn.Linear(
            model_kwargs["embedding_dim"],
            model_kwargs["embedding_dim"],
        )
        self.classification_head_linear_class = nn.Linear(
            model_kwargs["embedding_dim"],
            self.n_out_nodes,
        )

    def forward(self, x, mask):
        embeddings = F.relu(self.classification_head_linear_embed(x))
        embeddings_sum = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        logits = self.classification_head_linear_class(embeddings_sum)
        return logits


class ClassHeadFlatten(torch.nn.Module):
    def __init__(self, n_out_nodes, input_dim, dropout_rate=0.1, max_seq_len=128):
        super().__init__()

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim * max_seq_len, n_out_nodes)
        self.n_out_nodes = n_out_nodes

    def forward(self, x, mask):
        x = x * mask.unsqueeze(-1)
        x = self.in_norm(x)
        x = self.in_dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class BackboneClassificationLightning(L.LightningModule):
    """Backbone with classification head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        class_head_type: str = "summation",
        model_kwargs: dict = {},
        use_continuous_input: bool = True,
        scheduler_lightning_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        if self.hparams.use_continuous_input:
            self.backbone = BackboneTransformer(**model_kwargs)
            # initialize the continuous regression head (used as an additional task
            # to force the model to keep continuous feature information)
        else:
            self.backbone = BackboneModel(**model_kwargs)

        # initialize the model head
        if class_head_type == "summation":
            self.head = ClassificationHead(
                model_kwargs={
                    "n_out_nodes": model_kwargs["n_out_nodes"],
                    "embedding_dim": model_kwargs["embedding_dim"],
                }
            )
        elif class_head_type == "class_attention":
            self.head = ClassifierNormformer(
                input_dim=model_kwargs["embedding_dim"],
                hidden_dim=model_kwargs.get("class_head_hidden_dim", 128),
                num_heads=model_kwargs.get("class_head_num_heads", 8),
                num_class_blocks=model_kwargs.get("class_head_num_CA_blocks", 2),
                num_enc_blocks=model_kwargs.get("class_head_num_SA_blocks", 0),
                dropout_rate=model_kwargs.get("class_head_dropout_rate", 0.0),
                fc_params=model_kwargs.get("class_head_fc_params"),
                n_out_nodes=model_kwargs.get("n_out_nodes"),
                self_attention_model_class=model_kwargs.get(
                    "class_head_SA_model_class", "Normformer"
                ),
                cross_attention_model_class=model_kwargs.get(
                    "class_head_CA_model_class", "NormformerCrossBlock"
                ),
                identity_init=model_kwargs.get("class_head_identity_init", False),
            )
        elif class_head_type == "flatten":
            self.head = ClassHeadFlatten(
                n_out_nodes=model_kwargs["n_out_nodes"],
                input_dim=model_kwargs["embedding_dim"],
                max_seq_len=model_kwargs["max_sequence_len"],
                dropout_rate=0.1,
            )
        elif class_head_type == "linear_average_pool":
            self.head = ClassHeadLinAvgPool(
                input_dim=model_kwargs["embedding_dim"],
                intermediate_dim=model_kwargs.get("class_head_hidden_dim", 512),
                output_dim=model_kwargs["n_out_nodes"],
            )
        else:
            raise ValueError(f"Invalid class_head_type: {class_head_type}")

        # Handle class weights for imbalanced datasets
        class_weights = model_kwargs.get("class_weights", None)
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Using class weights: {class_weights}")
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.train_loss_history = []
        self.val_loss_history = []
        # this is just used to simplify the `self.log(...)` calls later on
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        logger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def forward(self, X, mask, x_jet=None):
        embeddings = self.backbone(X, mask, x_jet=x_jet)
        logits = self.head(embeddings, mask)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")
        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                load_backbone_weights(self, self.backbone_weights_path, strict=True)

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info(f"Epoch {self.trainer.current_epoch} started.")
        self.epoch_train_start_time = time.time()

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        X = batch["part_features"]
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None
        if self.hparams.model_kwargs.get("zero_padded_start_particle", False):
            X[:, 0] = 0.0
        mask = batch["part_mask"]
        jet_labels = batch["jet_type_labels"]
        # one-hot encode the labels
        logits = self.forward(X, mask, x_jet=X_jet)
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.head.n_out_nodes).float()
        loss = self.criterion(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "targets": labels,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)
        loss = model_step_output["loss"]
        logits = model_step_output["logits"]
        targets = model_step_output["targets"]

        preds = torch.softmax(logits, dim=1)
        if stage == "train":
            self.train_preds_list.append(preds.float().detach().cpu().numpy())
            self.train_labels_list.append(targets.float().detach().cpu().numpy())
            self.train_loss_history.append(loss.float().detach().cpu().numpy())
        elif stage == "val":
            self.val_preds_list.append(preds.float().detach().cpu().numpy())
            self.val_labels_list.append(targets.float().detach().cpu().numpy())
        elif stage == "test":
            self.test_preds_list.append(preds.float().detach().cpu().numpy())
            self.test_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(f"{stage}_loss", loss.item(), **self.log_dict)
        self.log(f"{stage}_acc", acc, **self.log_dict)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                logger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Rank: {self.global_rank}"
                )

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_preds_list = []
        self.test_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")
        return simple_optim_sched(self)

    # def configure_optimizers(self) -> Dict[str, Any]:
    #     """Configures optimizers and learning-rate schedulers to be used for training."""
    #     logger.info("`configure_optimizers` called.")
    #     if self.hparams.model_kwargs.get("keep_backbone_fixed", False):
    #         logger.info("--- Keeping backbone fixed. ---")
    #         optimizer = self.hparams.optimizer(
    #             [
    #                 {"params": self.module.parameters(), "lr": 0.0},
    #                 {"params": self.head.parameters()},
    #             ]
    #         )
    #     elif self.hparams.model_kwargs.get("keep_embedding_table_fixed", False):
    #         logger.info("--- Keeping embedding table fixed. ---")
    #         backbone_parameters_embedding_table = [
    #             params
    #             for name, params in self.module.named_parameters()
    #             if "embedding_table" in name
    #         ]
    #         backbone_parameters_other = [
    #             params
    #             for name, params in self.module.named_parameters()
    #             if "embedding_table" not in name
    #         ]
    #         optimizer = self.hparams.optimizer(
    #             [
    #                 {"params": backbone_parameters_embedding_table, "lr": 0.0},
    #                 {"params": backbone_parameters_other},
    #                 {"params": self.head.parameters()},
    #             ]
    #         )
    #     elif self.hparams.model_kwargs.get("backbone_lr", None) is not None:
    #         logger.info("--- Using different learning rate for backbone. ---")
    #         optimizer = self.hparams.optimizer(
    #             [
    #                 {
    #                     "params": self.module.parameters(),
    #                     "lr": self.hparams.model_kwargs["backbone_lr"],
    #                 },
    #                 {"params": self.head.parameters()},
    #             ]
    #         )
    #     else:
    #         optimizer = self.hparams.optimizer(params=self.parameters())
    #     if self.hparams.scheduler is not None:
    #         scheduler = self.hparams.scheduler(optimizer=optimizer)
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 **self.hparams.scheduler_lightning_kwargs,
    #             },
    #         }
    #     return {"optimizer": optimizer}


class BackboneMPMLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        verbose=False,
        use_continuous_input: bool = True,
        exclude_padded_values_from_loss: bool = True,
        scheduler_lightning_kwargs: dict = None,
        add_classifier_head: bool = False,
        detach_backbone_grad_before_class: bool = False,
        alpha_mpm_loss: float = 1.0,
        alpha_class_loss: float = 0,
        apply_positional_encoding_fix: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        if "token_dir" in model_kwargs and "token_dir" in kwargs:
            raise ValueError(
                "token_dir is defined in both model_kwargs and kwargs. Should only be specified in `model_kwargs`"
            )

        self.pylogger = get_pylogger(__name__)
        self.pylogger.info(f"Model kwargs: {model_kwargs}")
        self.token_dir = Path(model_kwargs["token_dir"])

        # this is just used to simplify the `self.log(...)` calls later on
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.vocab_size = model_kwargs["vocab_size"]

        if not self.hparams.use_continuous_input:
            raise ValueError("Only continuous input is supported for now in MPM.")

        self.backbone = BackboneTransformer(**model_kwargs)

        self.head = MPMHead(
            input_dim=model_kwargs["embedding_dim"],
            output_dim=self.vocab_size,
            hidden_dims=model_kwargs.get("mpm_head_hidden_dims", [128, 128]),
            transformer_cfg=model_kwargs.get("mpm_head_transformer_cfg", {}),
            apply_causal_mask=model_kwargs.get("mpm_head_apply_causal_mask", False),
        )

        if self.hparams.add_classifier_head:
            self.head_class = ClassHeadLinAvgPool(
                input_dim=model_kwargs["embedding_dim"],
                intermediate_dim=model_kwargs.get("class_head_hidden_dim", 512),
                output_dim=model_kwargs["n_out_nodes"],
            )

        logger.info(
            f"Preprocessing dict for continuous features: {self.backbone.particle_features_dict}"
        )
        self.load_vqvae_weights(**model_kwargs)

        logger.info(f"Preprocessing dict for VQ-VAE: {self.vqvae_pp_dict}")
        logger.info("Comparing the preprocessing dicts of the backbone and VQ-VAE model.")
        compare_two_pp_dicts(
            pp_dict_1=self.backbone.particle_features_dict,
            pp_dict_2=self.vqvae_pp_dict,
            ignore_features_not_present_in_second_dict=True,
        )

        # initialized masked feature vector as parameter, shape max_seq_len x embedding_dim
        self.masked_feature_vector = nn.Parameter(
            torch.randn(model_kwargs["max_sequence_len"], model_kwargs["embedding_dim"])
        )
        # positional embedding is a trainable parameter that is multiplied with
        # the first continuous feature (corresponding to ptrel)
        # self.positional_embedding_base_vector = nn.Parameter(
        #     torch.randn(model_kwargs["embedding_dim"])
        # )

        if self.hparams.model_kwargs.get("loss_weights", None) is not None:
            logger.warning(
                "Loss weights are not None, but not yet implemented. This will be ignored."
            )

        self.criterion = nn.CrossEntropyLoss()

        self.verbose = verbose
        self.gen_model_pp_dict = None

        self.train_loss_history = []

        # create empty lists to store the validation and test batches
        self.val_loss_list = []
        self.val_input_list = []
        self.val_mask_list = []
        self.val_valid_particle_mask_list = []
        self.val_valid_particle_but_masked_mask_list = []
        self.val_valid_particle_after_masking_mask_list = []
        self.val_input_list_jet = []
        self.val_token_pred_list = []
        self.val_token_target_list = []
        # same for test
        self.test_input_list = []
        self.test_mask_list = []
        self.test_valid_particle_mask_list = []
        self.test_valid_particle_but_masked_mask_list = []
        self.test_valid_particle_after_masking_mask_list = []
        self.test_input_list_jet = []
        self.test_token_pred_list = []
        self.test_token_target_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")

        self.pylogger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def load_vqvae_weights(self, **model_kwargs):
        """Load the VQ-VAE model weights.

        This is only used for the continuous input case.
        """

        # add the vqvae model to the lightning module but detach its parameters
        # the checkpoint is saved in the main directory of the data directory
        vqvae_ckpt = self.token_dir / "model_ckpt.ckpt"
        self.pylogger.info(f"Loading VQ-VAE model from {vqvae_ckpt}")
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt)
        self.max_sequence_len = model_kwargs["max_sequence_len"]

        self.pylogger.info(
            "Detaching the parameters of the VQ-VAE model from the gradient computation."
        )
        for param in self.vqvae_model.parameters():
            param.requires_grad = False

        # get the preprocessing dict used in the VQ-VAE model training
        vqvae_config_file = self.token_dir / "config.yaml"
        cfg = OmegaConf.load(vqvae_config_file)
        self.vqvae_pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict"])
        if "feature_dict_jet" in cfg.data.dataset_kwargs_common:
            if cfg.data.dataset_kwargs_common["feature_dict_jet"] is not None:
                self.vqvae_pp_dict_jet = OmegaConf.to_container(
                    cfg.data.dataset_kwargs_common["feature_dict_jet"]
                )
            else:
                self.vqvae_pp_dict_jet = None
        # the convention is that tokenized jets have vq-vae token + 1, which means
        # that 0 is the start token and num_codes + 1 is the stop token
        self.stop_token_value = self.vqvae_model.model.vq_kwargs["num_codes"] + 1

    def forward(
        self,
        x,
        valid_particle_mask=None,
        valid_particle_mask_corrupted=None,
        valid_particle_but_masked_mask=None,
        x_jet=None,
        return_logits_only=False,
    ):
        backbone_out = self.backbone(
            x,
            mask=valid_particle_mask_corrupted,
            x_jet=x_jet,
        )

        if self.hparams.add_classifier_head:
            backbone_out_for_class = backbone_out.clone()
            if self.hparams.detach_backbone_grad_before_class:
                backbone_out_for_class = backbone_out_for_class.detach()
            logits_class = self.head_class(backbone_out_for_class, mask=valid_particle_mask)
        else:
            logits_class = None
        # find indices where the valid_particle_but_masked_mask is 1
        indices = torch.nonzero(valid_particle_but_masked_mask, as_tuple=True)

        if self.hparams.apply_positional_encoding_fix:
            # calculate pT sorting of particles (first feature in the continuous input)
            # define pT proxy with invalid particles having pT -inf
            # (so that they are sorted to the end)
            proxy_pt = x[:, :, 0].clone()
            proxy_pt[valid_particle_mask == 0] = float("-inf")  # set invalid particles
            pt_sort_index = torch.argsort(proxy_pt, dim=1, descending=True)
            # print(f"x[0, :, 0]: {x[0, :, 0]}")
            # print(f"proxy_pt[0, :]: {proxy_pt[0, :]}")
            # print(f"pt_sort_index[0]: {pt_sort_index[0]}")
            # print(f"pt_sort_index[indices[0], indices[1]]: {pt_sort_index[indices[0], indices[1]]}")
            # set the backbone output for the masked particles to the masked feature vector
            backbone_out[indices] = self.masked_feature_vector[
                pt_sort_index[indices[0], indices[1]], :
            ]  # use the pT sorted indices to get the masked feature vector
            # print(f"Fist 3 index tuples: {indices[0, :3]}, {indices[1, :3]}")
            # print(f"Indices: {indices[0].shape}, {indices[1].shape}")
            # print(f"First 3 index tuples: {indices[0][:3]}, {indices[1][:3]}")
        else:
            # print(f"indices[1]: {indices[1]}")
            backbone_out[indices] = self.masked_feature_vector[indices[1]]
        # print(f"Backbone out of the first 3 indices: {backbone_out[indices][:3][..., :2]}")
        # print(backbone_out[indices])
        # add positional encoding to the particles which is the trainable
        # positional embedding multiplied with the first continuous feature
        # (which is the ptrel)
        # backbone_out = backbone_out + self.positional_embedding_base_vector * x[:, :, 1:2]
        logits_mpm = self.head(backbone_out, mask=valid_particle_mask)
        if self.verbose:
            self.pylogger.info("Logits shape: ", logits_mpm.shape)

        return logits_mpm, logits_class

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        """

        X = batch["part_features"].clone()
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None
        mask = batch["part_mask"].clone()

        targets = batch["part_labels"].clone().long()[:, :, 0]
        input = X

        mask = mask.int()
        valid_particle_mask = mask.clone()
        valid_particle_after_masking_mask = set_fraction_ones_to_zeros(
            mask,
            fraction=self.hparams.model_kwargs.get("mask_fraction", 0.15),
        )
        valid_particle_but_masked_mask = valid_particle_mask * (
            1 - valid_particle_after_masking_mask
        )

        logits_mpm, logits_class = self.forward(
            input,
            valid_particle_mask_corrupted=valid_particle_after_masking_mask,
            valid_particle_mask=valid_particle_mask,
            valid_particle_but_masked_mask=valid_particle_but_masked_mask,
            x_jet=X_jet,
        )
        targets_clone = targets.clone()

        if self.hparams.exclude_padded_values_from_loss:
            logits_mpm = fix_padded_logits(
                logits_mpm, valid_particle_but_masked_mask.bool(), factor=1e6
            )
            targets = targets * valid_particle_but_masked_mask

        # reshape the logits and targets to work with the loss function
        B, T, C = logits_mpm.shape
        logits_reshaped = logits_mpm.view(B * T, C)
        targets_reshaped = targets.contiguous().view(B * T)

        loss_masked_token_prediction = self.criterion(logits_reshaped, targets_reshaped)

        if self.hparams.add_classifier_head:
            labels_class = F.one_hot(
                batch["jet_type_labels"].squeeze(),
                num_classes=self.head_class.n_out_nodes,
            ).float()
            loss_class = self.criterion(logits_class, labels_class)
            loss = (
                self.hparams.alpha_mpm_loss * loss_masked_token_prediction
                + self.hparams.alpha_class_loss * loss_class
            )
        else:
            labels_class = None
            loss_class = torch.tensor(0)
            loss = loss_masked_token_prediction

        max_len_in_batch = logits_mpm.size(1)
        default_token_indices_to_calc = [0, 1, 2, 3, 10, 20, 30, 40, 50, 100]
        token_indices_to_calc = [
            index for index in default_token_indices_to_calc if index < max_len_in_batch
        ]

        acc_dict = calc_acc_from_logits(
            logits=logits_mpm,
            targets=targets,
            # only calculate accuracy for the particles that were masked
            mask=valid_particle_but_masked_mask,
            token_indices_to_calc=token_indices_to_calc,
        )

        return {
            "loss": loss,
            "loss_next_token_prediction": loss_masked_token_prediction,
            "X": X,
            "logits_reshaped": logits_reshaped,
            "targets_reshaped": targets_reshaped,
            "acc_dict": acc_dict,
            "logits": logits_mpm,
            "targets": targets_clone,
            "mask": mask,
            "valid_particle_mask": valid_particle_mask,
            "valid_particle_but_masked_mask": valid_particle_but_masked_mask,
            "valid_particle_after_masking_mask": valid_particle_after_masking_mask,
            "logits_class": logits_class,
            "targets_class": labels_class,
            "loss_class": loss_class,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)

        loss = model_step_output["loss"]
        loss_mtp = model_step_output["loss_next_token_prediction"]
        acc_dict = model_step_output["acc_dict"]

        if self.hparams.add_classifier_head:
            loss_class = model_step_output["loss_class"]
            acc_class = calc_accuracy(
                preds=torch.softmax(model_step_output["logits_class"], dim=1)
                .float()
                .detach()
                .cpu()
                .numpy(),
                labels=model_step_output["targets_class"].float().detach().cpu().numpy(),
            )
            self.log(f"{stage}_loss_class", loss_class.item(), **self.log_dict)
            self.log(f"{stage}_acc_class", acc_class, **self.log_dict)

        self.log(f"{stage}_loss", loss.item(), **self.log_dict)
        self.log(f"{stage}_loss_mtp", loss_mtp.item(), **self.log_dict)

        for key, value in acc_dict.items():
            self.log(f"{stage}_" + key, value, **self.log_dict)

        if stage in ["val", "test"]:
            self._collect_batch_data(batch, model_step_output, stage)

        return loss

    def _collect_batch_data(self, batch, model_step_output, stage: str) -> None:
        """Collect batch data for validation or test stages."""
        input_list = getattr(self, f"{stage}_input_list")
        mask_list = getattr(self, f"{stage}_mask_list")
        valid_particle_mask_list = getattr(self, f"{stage}_valid_particle_mask_list")
        valid_particle_but_masked_mask_list = getattr(
            self, f"{stage}_valid_particle_but_masked_mask_list"
        )
        valid_particle_after_masking_mask_list = getattr(
            self, f"{stage}_valid_particle_after_masking_mask_list"
        )
        token_pred_list = getattr(self, f"{stage}_token_pred_list")
        token_target_list = getattr(self, f"{stage}_token_target_list")

        input_list.append(batch["part_features"].float().detach().cpu().numpy())
        mask_list.append(batch["part_mask"].float().detach().cpu().numpy())
        valid_particle_mask_list.append(
            model_step_output["valid_particle_mask"].float().detach().cpu().numpy()
        )
        valid_particle_but_masked_mask_list.append(
            model_step_output["valid_particle_but_masked_mask"].float().detach().cpu().numpy()
        )
        valid_particle_after_masking_mask_list.append(
            model_step_output["valid_particle_after_masking_mask"].float().detach().cpu().numpy()
        )
        token_pred_list.append(
            torch.argmax(model_step_output["logits"], dim=-1).float().detach().cpu().numpy()
        )
        token_target_list.append(model_step_output["targets"].float().detach().cpu().numpy())
        if self.backbone.jet_features_input_dim > 0:
            getattr(self, f"{stage}_input_list_jet").append(
                batch["jet_features"].detach().cpu().numpy()
            )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    # on val/test epoch start, reset the lists to empty
    def on_validation_epoch_start(self) -> None:
        self.pylogger.info("`on_validation_epoch_start` called.")
        self.val_input_list = []
        self.val_mask_list = []
        self.val_valid_particle_mask_list = []
        self.val_valid_particle_but_masked_mask_list = []
        self.val_valid_particle_after_masking_mask_list = []
        self.val_input_list_jet = []
        self.val_token_pred_list = []
        self.val_token_target_list = []

    def on_test_epoch_start(self) -> None:
        self.pylogger.info("`on_test_epoch_start` called.")
        self.test_input_list = []
        self.test_mask_list = []
        self.test_valid_particle_mask_list = []
        self.test_valid_particle_but_masked_mask_list = []
        self.test_valid_particle_after_masking_mask_list = []
        self.test_input_list_jet = []
        self.test_token_pred_list = []
        self.test_token_target_list = []

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        self.pylogger.info("`configure_optimizers` called.")

        if self.hparams.model_kwargs.get("keep_embedding_table_fixed", False):
            logger.info("--- Keeping embedding table fixed. ---")
            backbone_parameters_embedding_table = [
                params
                for name, params in self.backbone.named_parameters()
                if "embedding_table" in name
            ]
            backbone_parameters_other = [
                params
                for name, params in self.backbone.named_parameters()
                if "embedding_table" not in name
            ]
            optimizer = self.hparams.optimizer(
                [
                    {"params": backbone_parameters_embedding_table, "lr": 0.0},
                    {"params": backbone_parameters_other},
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
                    **self.hparams.scheduler_lightning_kwargs,
                },
            }

        return {"optimizer": optimizer}


class BackboneDijetClassificationLightning(L.LightningModule):
    """Backbone with dijet classification head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        merge_strategy: str = "concat",
        class_head_type: str = "summation",
        model_kwargs: dict = {},
        use_continuous_input: bool = True,
        scheduler_lightning_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        self.merge_strategy = merge_strategy

        if self.hparams.use_continuous_input:
            self.backbone = BackboneTransformer(**model_kwargs)
            # initialize the continuous regression head (used as an additional task
            # to force the model to keep continuous feature information)
        else:
            self.backbone = BackboneModel(**model_kwargs)

        if merge_strategy == "concat":
            # Double the max_sequence_len
            model_kwargs["max_sequence_len"] *= 2

        elif merge_strategy == "attention":
            self.cross_attn_1_to_2 = nn.MultiheadAttention(
                    model_kwargs["embedding_dim"], 
                    model_kwargs["transformer_cfg"]["attn_cfg"]["num_heads"], 
                    dropout=model_kwargs["transformer_cfg"]["attn_cfg"]["dropout_rate"],
                    batch_first=True
                )
            self.cross_attn_2_to_1 = nn.MultiheadAttention(
                    model_kwargs["embedding_dim"], 
                    model_kwargs["transformer_cfg"]["attn_cfg"]["num_heads"], 
                    dropout=model_kwargs["transformer_cfg"]["attn_cfg"]["dropout_rate"],
                    batch_first=True
                )
            self.fusion_proj = nn.Linear(model_kwargs["embedding_dim"] * 2, model_kwargs["embedding_dim"])

        # initialize the model head
        if class_head_type == "summation":
            self.head = ClassificationHead(
                model_kwargs={
                    "n_out_nodes": model_kwargs["n_out_nodes"],
                    "embedding_dim": model_kwargs["embedding_dim"],
                }
            )
        elif class_head_type == "class_attention":
            self.head = ClassifierNormformer(
                input_dim=model_kwargs["embedding_dim"],
                hidden_dim=model_kwargs.get("class_head_hidden_dim", 128),
                num_heads=model_kwargs.get("class_head_num_heads", 8),
                num_class_blocks=model_kwargs.get("class_head_num_CA_blocks", 2),
                num_enc_blocks=model_kwargs.get("class_head_num_SA_blocks", 0),
                dropout_rate=model_kwargs.get("class_head_dropout_rate", 0.0),
                fc_params=model_kwargs.get("class_head_fc_params"),
                n_out_nodes=model_kwargs.get("n_out_nodes"),
                self_attention_model_class=model_kwargs.get(
                    "class_head_SA_model_class", "Normformer"
                ),
                cross_attention_model_class=model_kwargs.get(
                    "class_head_CA_model_class", "NormformerCrossBlock"
                ),
                identity_init=model_kwargs.get("class_head_identity_init", False),
            )
        elif class_head_type == "flatten":
            self.head = ClassHeadFlatten(
                n_out_nodes=model_kwargs["n_out_nodes"],
                input_dim=model_kwargs["embedding_dim"],
                max_seq_len=model_kwargs["max_sequence_len"],
                dropout_rate=0.1,
            )
        elif class_head_type == "linear_average_pool":
            self.head = ClassHeadLinAvgPool(
                input_dim=model_kwargs["embedding_dim"],
                intermediate_dim=model_kwargs.get("class_head_hidden_dim", 512),
                output_dim=model_kwargs["n_out_nodes"],
            )
        else:
            raise ValueError(f"Invalid class_head_type: {class_head_type}")

        # Handle class weights for imbalanced datasets
        class_weights = model_kwargs.get("class_weights", None)
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Using class weights: {class_weights}")
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.train_loss_history = []
        self.val_loss_history = []
        # this is just used to simplify the `self.log(...)` calls later on
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        logger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def forward(self, X1, mask1, X2, mask2, x_jet=None):
        embeddings_1 = self.backbone(X1, mask1, x_jet=x_jet)
        embeddings_2 = self.backbone(X2, mask2, x_jet=x_jet)

        # Convert masks to boolean for logical operations
        mask1_bool = mask1.bool()
        mask2_bool = mask2.bool()

        # Merge representations
        if self.merge_strategy == "concat":
            embeddings = torch.cat([embeddings_1, embeddings_2], dim=1) # Shape: (B, 256, 128)
            mask = torch.cat([mask1, mask2], dim=1)  # Shape: (B, 256)

        elif self.merge_strategy == "average":
            embeddings = (embeddings_1 + embeddings_2) / 2 # Shape: (B, 128, 128)
            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        elif self.merge_strategy == "weighted_sum":
            alpha = nn.Parameter(torch.tensor(0.5))
            embeddings = alpha * embeddings_1 + (1 - alpha) * embeddings_2 # Shape: (B, 128, 128)
            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        elif self.merge_strategy == "attention":
            # Jet1 attends to Jet2
            attn_1, _ = self.cross_attn_1_to_2(
                embeddings_1, embeddings_2, embeddings_2, key_padding_mask=~mask2_bool
            )
            
            # Jet2 attends to Jet1
            attn_2, _ = self.cross_attn_2_to_1(
                embeddings_2, embeddings_1, embeddings_1, key_padding_mask=~mask1_bool
            )

            # Combine
            embeddings = torch.cat([attn_1, attn_2], dim=-1)
            embeddings = self.fusion_proj(embeddings) # (B, 128, 128)

            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        mask = mask.to(torch.float32)
        logits = self.head(embeddings, mask)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")
        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                load_backbone_weights(self, self.backbone_weights_path, strict=True)

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info(f"Epoch {self.trainer.current_epoch} started.")
        self.epoch_train_start_time = time.time()

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # Extract Dijet features and labels from the batch
        X1 = batch["part_features"]
        X2 = batch["part_features_jet2"]
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None
        if self.hparams.model_kwargs.get("zero_padded_start_particle", False):
            X1[:, 0] = 0.0
            X2[:, 0] = 0.0
        mask1 = batch["part_mask"]
        mask2 = batch["part_mask_jet2"]
        jet_labels = batch["jet_type_labels"]
        # one-hot encode the labels
        logits = self.forward(X1, mask1, X2, mask2, x_jet=X_jet)
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.head.n_out_nodes).float()
        loss = self.criterion(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "targets": labels,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)
        loss = model_step_output["loss"]
        logits = model_step_output["logits"]
        targets = model_step_output["targets"]

        preds = torch.softmax(logits, dim=1)
        if stage == "train":
            self.train_preds_list.append(preds.float().detach().cpu().numpy())
            self.train_labels_list.append(targets.float().detach().cpu().numpy())
            self.train_loss_history.append(loss.float().detach().cpu().numpy())
        elif stage == "val":
            self.val_preds_list.append(preds.float().detach().cpu().numpy())
            self.val_labels_list.append(targets.float().detach().cpu().numpy())
        elif stage == "test":
            self.test_preds_list.append(preds.float().detach().cpu().numpy())
            self.test_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(f"{stage}_loss", loss.item(), **self.log_dict)
        self.log(f"{stage}_acc", acc, **self.log_dict)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                logger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Rank: {self.global_rank}"
                )

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_preds_list = []
        self.test_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")
        return simple_optim_sched(self)
    

class BackboneAachenClassificationLightning(L.LightningModule):
    """Backbone with dijet classification head using Aachen anomaly detection architecture."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        merge_strategy: str = "concat",
        model_kwargs: dict = {},
        use_continuous_input: bool = True,
        scheduler_lightning_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        self.merge_strategy = merge_strategy

        if self.hparams.use_continuous_input:
            self.backbone = BackboneTransformer(**model_kwargs)
            # initialize the continuous regression head (used as an additional task
            # to force the model to keep continuous feature information)
        else:
            self.backbone = BackboneModel(**model_kwargs)

        if merge_strategy == "concat":
            # Double the max_sequence_len
            model_kwargs["max_sequence_len"] *= 2

        elif merge_strategy == "attention":
            self.cross_attn_1_to_2 = nn.MultiheadAttention(
                    model_kwargs["embedding_dim"], 
                    model_kwargs["transformer_cfg"]["attn_cfg"]["num_heads"], 
                    dropout=model_kwargs["transformer_cfg"]["attn_cfg"]["dropout_rate"],
                    batch_first=True
                )
            self.cross_attn_2_to_1 = nn.MultiheadAttention(
                    model_kwargs["embedding_dim"], 
                    model_kwargs["transformer_cfg"]["attn_cfg"]["num_heads"], 
                    dropout=model_kwargs["transformer_cfg"]["attn_cfg"]["dropout_rate"],
                    batch_first=True
                )
            self.fusion_proj = nn.Linear(model_kwargs["embedding_dim"] * 2, model_kwargs["embedding_dim"])

        # Aachen anomaly detection architecture
        self.encoder_layer = TransformerEncoderLayer(
            d_model=model_kwargs["class_head_hidden_dim"],
            nhead=model_kwargs["class_head_num_heads"],
            dim_feedforward=model_kwargs["class_head_hidden_dim"],
            dropout=model_kwargs["class_head_dropout_rate"],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.head_transformer = TransformerEncoder(
            self.encoder_layer, 
            num_layers=model_kwargs["class_head_num_CA_blocks"]
        )
        
        # Project backbone embeddings to head dimension if different
        if model_kwargs["embedding_dim"] != model_kwargs["class_head_hidden_dim"]:
            self.embedding_projection = nn.Linear(
                model_kwargs["embedding_dim"],
                model_kwargs["class_head_hidden_dim"]
            )
        else:
            self.embedding_projection = nn.Identity()
        
        # Final classification (binary: 1 output node)
        self.head = nn.Linear(model_kwargs["class_head_hidden_dim"], 1)
        self.n_out_nodes = 2  # Binary classification

        # Handle class weights for imbalanced datasets
        class_weights = model_kwargs.get("class_weights", None)
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Using class weights: {class_weights}")
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights[1] if class_weights is not None else None
        )

        self.train_loss_history = []
        self.val_loss_history = []
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        logger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def forward(self, X1, mask1, X2, mask2, x_jet=None):
        embeddings_1 = self.backbone(X1, mask1, x_jet=x_jet)
        embeddings_2 = self.backbone(X2, mask2, x_jet=x_jet)

        # Convert masks to boolean for logical operations
        mask1_bool = mask1.bool()
        mask2_bool = mask2.bool()

        # Merge representations
        if self.merge_strategy == "concat":
            embeddings = torch.cat([embeddings_1, embeddings_2], dim=1) # Shape: (B, 256, 128)
            mask = torch.cat([mask1, mask2], dim=1)  # Shape: (B, 256)

        elif self.merge_strategy == "average":
            embeddings = (embeddings_1 + embeddings_2) / 2 # Shape: (B, 128, 128)
            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        elif self.merge_strategy == "weighted_sum":
            alpha = nn.Parameter(torch.tensor(0.5))
            embeddings = alpha * embeddings_1 + (1 - alpha) * embeddings_2 # Shape: (B, 128, 128)
            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        elif self.merge_strategy == "attention":
            # Jet1 attends to Jet2
            attn_1, _ = self.cross_attn_1_to_2(
                embeddings_1, embeddings_2, embeddings_2, key_padding_mask=~mask2_bool
            )
            
            # Jet2 attends to Jet1
            attn_2, _ = self.cross_attn_2_to_1(
                embeddings_2, embeddings_1, embeddings_1, key_padding_mask=~mask1_bool
            )

            # Combine
            embeddings = torch.cat([attn_1, attn_2], dim=-1)
            embeddings = self.fusion_proj(embeddings) # (B, 128, 128)

            mask = mask1_bool & mask2_bool # Shape: (B, 128)

        mask = mask.to(torch.float32)
        # Project embeddings if needed
        embeddings = self.embedding_projection(embeddings)
        
        # Pass through transformer encoder
        # Create padding mask for transformer (True = ignore)
        src_key_padding_mask = ~mask.bool()
        embeddings = self.head_transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global average pooling over valid particles
        embeddings_masked = embeddings * mask.unsqueeze(-1)
        embeddings_sum = embeddings_masked.sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        embeddings_avg = embeddings_sum / valid_count
        
        # Final classification
        logits = self.head(embeddings_avg).squeeze(-1)  # (B,)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")
        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                load_backbone_weights(self, self.backbone_weights_path, strict=True)

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info(f"Epoch {self.trainer.current_epoch} started.")
        self.epoch_train_start_time = time.time()

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # Extract Dijet features and labels from the batch
        X1 = batch["part_features"]
        X2 = batch["part_features_jet2"]
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None

        if self.hparams.model_kwargs.get("zero_padded_start_particle", False):
            X1[:, 0] = 0.0
            X2[:, 0] = 0.0

        mask1 = batch["part_mask"]
        mask2 = batch["part_mask_jet2"]
        jet_labels = batch["jet_type_labels"].squeeze()

        # Forward pass
        logits = self.forward(X1, mask1, X2, mask2, x_jet=X_jet)

        # Binary classification with BCEWithLogitsLoss
        # logits shape: (B,), labels shape: (B,)
        labels = jet_labels.float()
        loss = self.criterion(logits, labels)
        
        # For metrics, convert to 2-class probabilities
        probs_signal = torch.sigmoid(logits)
        probs = torch.stack([1 - probs_signal, probs_signal], dim=1)
        targets = F.one_hot(jet_labels.long(), num_classes=2).float()
            
        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
            "probs": probs,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)
        loss = model_step_output["loss"]
        logits = model_step_output["logits"]
        targets = model_step_output["targets"]

        # Use precomputed probs from model_step (already in correct shape)
        preds = model_step_output["probs"]
        if stage == "train":
            self.train_preds_list.append(preds.float().detach().cpu().numpy())
            self.train_labels_list.append(targets.float().detach().cpu().numpy())
            self.train_loss_history.append(loss.float().detach().cpu().numpy())
        elif stage == "val":
            self.val_preds_list.append(preds.float().detach().cpu().numpy())
            self.val_labels_list.append(targets.float().detach().cpu().numpy())
        elif stage == "test":
            self.test_preds_list.append(preds.float().detach().cpu().numpy())
            self.test_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(f"{stage}_loss", loss.item(), **self.log_dict)
        self.log(f"{stage}_acc", acc, **self.log_dict)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                logger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Rank: {self.global_rank}"
                )

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_preds_list = []
        self.test_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")
        return simple_optim_sched(self)