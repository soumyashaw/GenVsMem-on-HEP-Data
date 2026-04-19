"""Callback for evaluating the MPM model."""

import gc
import math
import os

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import vector
from lightning.pytorch.utilities import rank_zero_only

import gabbro.plotting.utils as plot_utils
from gabbro.metrics.jet_substructure import JetSubstructure
from gabbro.metrics.utils import calc_quantiled_kl_divergence_for_dict
from gabbro.plotting.feature_plotting import plot_features
from gabbro.plotting.utils import DEFAULT_LABELS
from gabbro.utils.arrays import (
    ak_abs,
    ak_mean,
    ak_pad,
    ak_select_and_preprocess,
    ak_subtract,
    ak_to_np_stack,
    np_to_ak,
    p4s_from_ptetaphimass,
)
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()


class MPMEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
        n_val_gen_jets: int = 10,
        n_final_gen_jets: int = 10,
        starting_at_epoch: int = 0,
        every_n_epochs: int = 1,
        batch_size_for_generation: int = 512,
        class_token: int = None,
        seed: int = 42,
        bins_dict: dict = {},
    ):
        """Callback for evaluating the tokenization of particles.

        Parameters
        ----------
        image_path : str
            Path to save the images to. If None, the images are saved to the
            default_root_dir of the trainer.
        image_filetype : str
            Filetype to save the images as. Default is "png".
        no_trainer_info_in_filename : bool
            If True, the filename of the images will not contain the epoch and
            global step information. Default is False.
        save_result_arrays : bool
            If True, the results are saved as parquet file. Default is None.
        n_val_gen_jets : int
            Number of validation jets to generate. Default is 10.
        n_final_gen_jets : int
            Number of jets to generate at the end of training (or if not training). Default is 10.
        starting_at_epoch : int
            Start evaluating the model at this epoch. Default is 0.
        every_n_epochs : int
            Evaluate the model every n epochs. Default is 1.
        batch_size_for_generation : int
            Batch size for generating the jets. Default is 512.
        """
        super().__init__()
        self.comet_logger = None
        self.pylogger = get_pylogger(__name__)
        self.image_path = image_path
        self.n_val_gen_jets = n_val_gen_jets
        self.n_final_gen_jets = n_final_gen_jets
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays
        self.every_n_epochs = every_n_epochs
        self.starting_at_epoch = starting_at_epoch
        self.batch_size_for_generation = batch_size_for_generation
        self.class_token = class_token
        self.seed = seed
        self.bins_dict = {
            key: np.linspace(value["start"], value["stop"], value["num"])
            for key, value in bins_dict.items()
        }

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.pylogger.info("`on_train_start` called.")
        self.pylogger.info("Setting up the logger with the correct rank.")
        self.pylogger = get_pylogger(__name__, rank=trainer.global_rank)
        self.pylogger.info("Logger set up.")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # seed everything with lightning
        L.seed_everything(self.seed)

        if trainer.current_epoch < self.starting_at_epoch:
            self.pylogger.info(
                "Skipping generation. Starting evaluating with this callback"
                f" at epoch {self.starting_at_epoch}."
            )
            return None
        if trainer.current_epoch % self.every_n_epochs != 0:
            self.pylogger.info(
                f"Skipping generation. Only evaluating every {self.every_n_epochs} epochs."
            )
            return None
        if len(pl_module.val_input_list) == 0:
            self.pylogger.warning("No validation data available. Skipping generation.")
            return None

        return self.plot(trainer, pl_module, stage="val")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.plot(trainer, pl_module, stage="test")

    @rank_zero_only
    def plot(self, trainer, pl_module, stage):
        plot_utils.set_mpl_style()

        if hasattr(pl_module, "head_mpm"):  # new style in mulihead models
            if pl_module.head_mpm is None:
                self.pylogger.warning(
                    "No head_mpm found in the model. Skipping mpm evaluation and plotting."
                )
                return None

        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        if stage == "val":
            input_features_list = pl_module.val_input_list
            valid_particle_mask_list = pl_module.val_valid_particle_mask_list
            valid_particle_after_masking_mask_list = (
                pl_module.val_valid_particle_after_masking_mask_list
            )
            valid_particle_but_masked_mask_list = pl_module.val_valid_particle_but_masked_mask_list
            token_preds_list = (
                pl_module.val_token_pred_list
                if hasattr(pl_module, "val_token_pred_list")
                else pl_module.val_mpm_pred_list
            )
            token_targets_list = (
                pl_module.val_token_target_list
                if hasattr(pl_module, "val_token_target_list")
                else pl_module.val_mpm_target_list
            )
            jet_features_list = pl_module.val_input_list_jet
        elif stage == "test":
            input_features_list = pl_module.test_input_list
            valid_particle_mask_list = pl_module.test_valid_particle_mask_list
            valid_particle_after_masking_mask_list = (
                pl_module.test_valid_particle_after_masking_mask_list
            )
            valid_particle_but_masked_mask_list = (
                pl_module.test_valid_particle_but_masked_mask_list
            )
            token_preds_list = (
                pl_module.test_token_pred_list
                if hasattr(pl_module, "test_token_pred_list")
                else pl_module.test_mpm_pred_list
            )
            token_targets_list = (
                pl_module.test_token_target_list
                if hasattr(pl_module, "test_token_target_list")
                else pl_module.test_mpm_target_list
            )
            jet_features_list = pl_module.test_input_list_jet
        else:
            raise ValueError(f"Unknown stage: {stage}")

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)

        input_features = np.concatenate(input_features_list)
        jet_features = np.concatenate(jet_features_list)
        valid_particle_mask = np.concatenate(valid_particle_mask_list)
        valid_particle_after_masking_mask = np.concatenate(valid_particle_after_masking_mask_list)
        valid_particle_but_masked_mask = np.concatenate(valid_particle_but_masked_mask_list)
        token_preds = np.concatenate(token_preds_list)
        token_targets = np.concatenate(token_targets_list)

        # if lightning module has attribute `head_mpm`, then it's the new multi-head style
        # and we have a zero-padded particle at the beginning that we want to remove
        if hasattr(pl_module, "head_mpm"):
            input_features = input_features[:, 1:, :]
            valid_particle_mask = valid_particle_mask[:, 1:]
            valid_particle_after_masking_mask = valid_particle_after_masking_mask[:, 1:]
            valid_particle_but_masked_mask = valid_particle_but_masked_mask[:, 1:]
            token_preds = token_preds[:, 1:]
            token_targets = token_targets[:, 1:]

        print(f"input_features.shape: {input_features.shape}")
        print(f"jet_features.shape: {jet_features.shape}")
        print(f"valid_particle_mask.shape: {valid_particle_mask.shape}")
        print(f"valid_particle_but_masked_mask.shape: {valid_particle_but_masked_mask.shape}")
        print(
            f"valid_particle_after_masking_mask.shape: {valid_particle_after_masking_mask.shape}"
        )
        print(f"token_preds.shape: {token_preds.shape}")
        print(f"token_targets.shape: {token_targets.shape}")

        feature_dict = pl_module.backbone.particle_features_dict
        # convert to ak arrays
        input_features_ak = np_to_ak(
            input_features,
            names=list(feature_dict.keys()),
            mask=valid_particle_mask,
        )
        input_features_masked_ak = np_to_ak(
            input_features,
            names=list(feature_dict.keys()),
            mask=valid_particle_but_masked_mask,
        )
        input_features_not_masked_ak = np_to_ak(
            input_features,
            names=list(feature_dict.keys()),
            mask=valid_particle_after_masking_mask,
        )
        input_features_ak = ak_select_and_preprocess(
            ak_array=input_features_ak, pp_dict=feature_dict, inverse=True
        )
        input_features_masked_ak = ak_select_and_preprocess(
            ak_array=input_features_masked_ak, pp_dict=feature_dict, inverse=True
        )
        input_features_not_masked_ak = ak_select_and_preprocess(
            ak_array=input_features_not_masked_ak, pp_dict=feature_dict, inverse=True
        )
        # p4s = p4s_from_ptetaphimass(ak_arr=input_features_ak)
        p4s_masked = p4s_from_ptetaphimass(ak_arr=input_features_masked_ak)
        p4s_not_masked = p4s_from_ptetaphimass(ak_arr=input_features_not_masked_ak)

        # set token preds to 1 where it's 0 to avoid errors
        # TODO: this is just a workaround for now --> fix this (should probably use
        # the non-shifted tokens right away)
        token_preds[token_preds == 0] = 1
        token_targets[token_targets == 0] = 1

        # set the values of the non-masked particles to the target ones
        token_preds[valid_particle_after_masking_mask == 1] = token_targets[
            valid_particle_after_masking_mask == 1
        ]

        # subtract the token ids by 1 since we use the same token-ids as in the
        # generative case and thus we have to correct them by the shift we introduced
        # with the start tokens
        token_preds_ak = np_to_ak(
            token_preds[:, :, None] - 1,
            names=["token_id"],
            mask=valid_particle_mask,
        )
        token_targets_ak = np_to_ak(
            token_targets[:, :, None] - 1,
            names=["token_id"],
            mask=valid_particle_mask,
        )

        # reconstruct the predicted tokens
        vqvae_model = pl_module.vqvae_model
        vqvae_model.eval()
        reco_tokens_pred = vqvae_model.reconstruct_ak_tokens(
            tokens_ak=token_preds_ak,
            pp_dict=pl_module.vqvae_pp_dict,
        )
        reco_tokens_target = vqvae_model.reconstruct_ak_tokens(
            tokens_ak=token_targets_ak,
            pp_dict=pl_module.vqvae_pp_dict,
        )
        # convert both to numpy arrays
        maxlen = token_preds.shape[1]
        reco_tokens_pred_np = ak_to_np_stack(
            ak_pad(reco_tokens_pred, maxlen=maxlen), names=reco_tokens_pred.fields
        )
        _, reco_tokens_mask = ak_pad(reco_tokens_pred, return_mask=True, maxlen=maxlen)
        reco_tokens_target_np = ak_to_np_stack(
            ak_pad(reco_tokens_target, maxlen=maxlen), names=reco_tokens_target.fields
        )
        # back to ak arrays
        reco_tokens_pred = np_to_ak(
            reco_tokens_pred_np,
            names=reco_tokens_pred.fields,
            mask=reco_tokens_mask,
        )
        reco_tokens_target = np_to_ak(
            reco_tokens_target_np,
            names=reco_tokens_target.fields,
            mask=reco_tokens_mask,
        )

        print(f"Fields of the reconstructed tokens: {reco_tokens_pred.fields}")

        p4s_reco_pred = p4s_from_ptetaphimass(reco_tokens_pred)
        p4s_reco_target = p4s_from_ptetaphimass(reco_tokens_target)

        self.pylogger.info("Calculating the jet substructure.")
        # substructure_real = JetSubstructure(p4s[ak.num(p4s) > 3])
        substructure_masked = JetSubstructure(p4s_masked[ak.num(p4s_masked) > 3])
        substructure_not_masked = JetSubstructure(p4s_not_masked[ak.num(p4s_not_masked) > 3])
        substructure_reco_pred = JetSubstructure(p4s_reco_pred[ak.num(p4s_reco_pred) > 3])
        substructure_reco_target = JetSubstructure(p4s_reco_target[ak.num(p4s_reco_target) > 3])

        self.pylogger.info("Plotting the jet substructure.")
        # line below is the same as reconstructing the target tokens
        # substructure_real_ak = substructure_real.get_substructure_as_ak_array()
        substructure_masked_ak = substructure_masked.get_substructure_as_ak_array()
        substructure_not_masked_ak = substructure_not_masked.get_substructure_as_ak_array()
        substructure_reco_pred_ak = substructure_reco_pred.get_substructure_as_ak_array()
        substructure_reco_target_ak = substructure_reco_target.get_substructure_as_ak_array()
        substructure_reco_target_diff_ak = ak_subtract(
            substructure_reco_target_ak, substructure_reco_pred_ak
        )
        # calculate mean error and mean absolute error of diff
        substructure_diff_meanabserr_ak = ak_mean(ak_abs(substructure_reco_target_diff_ak))
        substructure_diff_meanerr_ak = ak_mean(substructure_reco_target_diff_ak)

        names_labels_dict_for_plotting = {
            "jet_pt": "Jet $p_T$ [GeV]",
            "jet_eta": "Jet $\\eta$",
            "jet_phi": "Jet $\\phi$",
            "jet_mass": "Jet mass [GeV]",
            "tau32": "$\\tau_{32}$",
            "tau21": "$\\tau_{21}$",
            "jet_n_constituents": "Number of constituents",
        }

        bins_dict = {
            "jet_pt": np.linspace(0, 1200, 91),
            "jet_eta": np.linspace(-0.1, 0.1, 70),
            "jet_phi": np.linspace(-0.05, 0.05, 70),
            "jet_mass": np.linspace(0, 250, 70),
            "tau32": np.linspace(0, 1.2, 70),
            "tau21": np.linspace(0, 1.2, 70),
            "jet_n_constituents": np.linspace(-0.5, 128.5, 130),
        }
        fig, axarr = plot_features(
            ak_array_dict={
                "All": substructure_reco_target_ak,
                "Masked": substructure_masked_ak,
                "Reconstructed": substructure_reco_pred_ak,
                "Not masked": substructure_not_masked_ak,
            },
            names=names_labels_dict_for_plotting,
            bins_dict=bins_dict,
            flatten=False,
            ax_rows=2,
            legend_kwargs={"loc": "upper center", "ncol": 2},
            ax_size=(3.2, 2),
            ratio=True,
            decorate_ax_kwargs={"yscale": 1.6},
        )
        image_filename_jet_features = (
            f"{plot_dir}/{stage}_epoch{trainer.current_epoch}_gstep{trainer.global_step}_"
            f"jet_features.{self.image_filetype}"
        )
        fig.savefig(image_filename_jet_features)
        self.pylogger.info(f"Saved jet features plot to {image_filename_jet_features}")

        bins_dict = {
            "jet_pt": np.linspace(-15, 15, 91),
            "jet_eta": np.linspace(-0.05, 0.05, 70),
            "jet_phi": np.linspace(-0.05, 0.05, 70),
            "jet_mass": np.linspace(-15, 15, 70),
            "tau32": np.linspace(-0.1, 0.1, 70),
            "tau21": np.linspace(-0.1, 0.1, 70),
            "jet_n_constituents": np.linspace(-4.5, 4.5, 10),
        }

        # plot the difference between the target and the prediction on jet level
        fig, axarr = plot_features(
            ak_array_dict={"Reconstructed - Target": substructure_reco_target_diff_ak},
            names={
                name: f"{label} (reco. - target)"
                for name, label in names_labels_dict_for_plotting.items()
            },
            bins_dict=bins_dict,
            flatten=False,
            ax_rows=2,
            ax_size=(3.2, 2),
            decorate_ax_kwargs={"yscale": 1.6},
        )
        image_filename_jet_features_diff = image_filename_jet_features.replace(
            "jet_features", "jet_features_diff"
        )
        fig.savefig(image_filename_jet_features_diff)
        self.pylogger.info(f"Saved jet features diff plot to {image_filename_jet_features_diff}")

        # ----------------------------------------------------------------------
        # Particle features
        # -----------------
        names_to_plot = input_features_ak.fields
        print(f"Fields of the input features: {names_to_plot}")
        # remote part_token_id if it is present
        names_to_plot = [name for name in names_to_plot if name != "part_token_id"]

        # remove ["part_px", "part_py", "part_pz", "part_energy"] if they are present
        names_to_plot = [
            name
            for name in names_to_plot
            if name not in ["part_px", "part_py", "part_pz", "part_energy"]
        ]
        bins_dict_for_particles = {
            "part_pt": np.linspace(0, 500, 100),
            "part_etarel": np.linspace(-0.9, 0.9, 70),
            "part_phirel": np.linspace(-0.9, 0.9, 70),
            "part_mass": np.linspace(-0.1, 1.0, 111),
            "part_ptrel": np.linspace(-0.1, 1.1, 121),
            "part_charge": np.linspace(-1.1, 1.1, 56),
            "part_isChargedHadron": np.linspace(-0.1, 1.1, 61),
            "part_isNeutralHadron": np.linspace(-0.1, 1.1, 61),
            "part_isPhoton": np.linspace(-0.1, 1.1, 61),
            "part_isElectron": np.linspace(-0.1, 1.1, 61),
            "part_isMuon": np.linspace(-0.1, 1.1, 61),
            "part_d0val": np.linspace(-0.1, 0.1, 101),
            "part_dzval": np.linspace(-0.1, 0.1, 101),
            "part_d0err": np.linspace(-0, 0.1, 100),
            "part_dzerr": np.linspace(-0, 0.1, 100),
        }
        fig, axarr = plot_features(
            ak_array_dict={
                "All": input_features_ak,
                "Masked": input_features_masked_ak,
                "Reconstructed": reco_tokens_pred,
                "Not masked": input_features_not_masked_ak,
            },
            names={name: DEFAULT_LABELS.get(name, name) for name in names_to_plot},
            # legend_only_on=0,
            legend_kwargs={"loc": "upper center", "ncol": 2},
            ax_rows=math.ceil(len(input_features_ak.fields) / 4),
            ax_size=(3.2, 2),
            ratio=True,
            bins_dict=bins_dict_for_particles,
            logscale_features=["part_pt"],
            decorate_ax_kwargs={"yscale": 1.6},
        )
        image_filename_particle_features = image_filename_jet_features.replace(
            "jet_features", "particle_features"
        )
        fig.savefig(image_filename_particle_features)
        self.pylogger.info(f"Saved particle features plot to {image_filename_particle_features}")

        # calculate the difference of the particle features
        particle_reco_diff_pred_to_target = ak_subtract(reco_tokens_pred, reco_tokens_target)
        particle_reco_diff_pred_to_target_meanabserr = ak_mean(
            ak_abs(particle_reco_diff_pred_to_target)
        )
        particle_reco_diff_pred_to_target_meanerr = ak_mean(particle_reco_diff_pred_to_target)

        bins_dict_for_particles = {
            "part_pt": np.linspace(-1, 1, 100),
            "part_etarel": np.linspace(-0.02, 0.02, 70),
            "part_phirel": np.linspace(-0.02, 0.02, 70),
        }

        # plot those distributions
        fig, axarr = plot_features(
            ak_array_dict={"Reconstructed - Target": particle_reco_diff_pred_to_target},
            names={
                name: f"{DEFAULT_LABELS.get(name, name)} (reco. - target)"
                for name in names_to_plot
            },
            bins_dict=bins_dict_for_particles,
            flatten=True,
            ax_rows=math.ceil(len(input_features_ak.fields) / 4),
            ax_size=(3.2, 2),
            decorate_ax_kwargs={"yscale": 1.6},
        )
        image_filename_particle_features_diff = image_filename_jet_features.replace(
            "jet_features", "particle_features_diff"
        )
        fig.savefig(image_filename_particle_features_diff)
        self.pylogger.info(
            f"Saved particle features diff plot to {image_filename_particle_features_diff}"
        )
        # ----------------------------------------------------------------------
        # eta-phi scatter plots of 9 examples
        # -----------------
        n_examples = 9
        n_cols = 3
        n_rows = math.ceil(n_examples / n_cols)
        fig, axarr = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.9 * n_cols, 3.7 * n_rows),
            layout="constrained",
        )
        for i in range(n_examples):
            ax = axarr[i // n_cols, i % n_cols]
            # eta-phi scatter plot, dot scaled with pt value
            ax.scatter(
                reco_tokens_target.part_etarel[i],
                reco_tokens_target.part_phirel[i],
                s=2 * reco_tokens_target.part_pt[i],
                label="Reco. target tokens",
                alpha=1,
                # don't fill the markers
                edgecolors="black",
                facecolors="none",
                linewidth=0.8,
            )
            ax.scatter(
                reco_tokens_pred.part_etarel[i],
                reco_tokens_pred.part_phirel[i],
                s=2 * reco_tokens_pred.part_pt[i],
                label="Reco. tokens\n(pred. and non-masked)",
                alpha=0.3,
                color="C3",
            )
            ax.scatter(
                input_features_masked_ak.part_etarel[i],
                input_features_masked_ak.part_phirel[i],
                s=2 * input_features_masked_ak.part_pt[i],
                label="Input masked",
                alpha=0.8,
                marker="+",
                color="C0",
            )
            ax.scatter(
                input_features_not_masked_ak.part_etarel[i],
                input_features_not_masked_ak.part_phirel[i],
                s=2 * input_features_not_masked_ak.part_pt[i],
                label="Input not masked",
                alpha=0.8,
                marker="+",
                color="C1",
            )
            ax.set_xlabel(DEFAULT_LABELS.get("part_etarel", "part_etarel"))
            ax.set_ylabel(DEFAULT_LABELS.get("part_phirel", "part_phirel"))
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 1.1)
            # draw a circle to indicate the jet radius of 0.8
            circle = plt.Circle(
                (0, 0), 0.8, color="black", fill=False, linestyle="--", linewidth=1, alpha=0.5
            )
            ax.add_artist(circle)
            # make legend but with boxes as handles instead of dots (but the colors are the same)
            handles, labels = ax.get_legend_handles_labels()
            handles = [
                ax.scatter([], [], s=50, alpha=0.5, label=label, color=h.get_edgecolor()[0])
                for h, label in zip(handles, labels)
            ]
            ax.legend(handles, labels, loc="upper center", ncol=2, fontsize=7)

        fig.tight_layout()
        image_filename_eta_phi_scatter = image_filename_jet_features.replace(
            "jet_features", "eta_phi_scatter"
        )
        fig.savefig(image_filename_eta_phi_scatter)
        self.pylogger.info(f"Saved eta-phi scatter plot to {image_filename_eta_phi_scatter}")

        # calculate the kld between the target and the prediction
        kld_dict = calc_quantiled_kl_divergence_for_dict(
            dict_reference=substructure_reco_target_ak,
            dict_approx=substructure_reco_pred_ak,
            names=list(names_labels_dict_for_plotting.keys()),
            n_bins=50,
            return_zero_if_nan_or_inf=True,
        )
        self.pylogger.info(f"KLD values: {kld_dict}")

        if self.comet_logger is not None:
            # log the KLD values
            for key, value in kld_dict.items():
                self.comet_logger.log_metric(f"{stage}_kld_{key}", value, step=trainer.global_step)
            for field in substructure_diff_meanerr_ak.keys():
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{field}",
                    substructure_diff_meanerr_ak[field],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_abs_err_{field}",
                    substructure_diff_meanabserr_ak[field],
                    step=trainer.global_step,
                )
            for field in particle_reco_diff_pred_to_target_meanerr.keys():
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{field}",
                    particle_reco_diff_pred_to_target_meanerr[field],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_abs_err_{field}",
                    particle_reco_diff_pred_to_target_meanabserr[field],
                    step=trainer.global_step,
                )
            for filename in [
                image_filename_jet_features,
                image_filename_jet_features_diff,
                image_filename_particle_features,
                image_filename_particle_features_diff,
                image_filename_eta_phi_scatter,
            ]:
                self.comet_logger.log_image(
                    filename,
                    name=filename.split("/")[-1],
                    step=trainer.global_step,
                )
