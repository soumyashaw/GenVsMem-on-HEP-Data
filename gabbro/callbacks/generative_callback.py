"""Callback for evaluating the generative token model."""

import gc
import math
import os
from pathlib import Path

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import vector
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

import gabbro.plotting.utils as plot_utils
from gabbro.data.data_tokenization import reconstruct_jetclass_file
from gabbro.metrics.jet_substructure import JetSubstructure
from gabbro.metrics.utils import calc_quantiled_kl_divergence_for_dict
from gabbro.plotting.feature_plotting import plot_features
from gabbro.plotting.utils import DEFAULT_LABELS
from gabbro.utils.arrays import np_to_ak, p4s_from_ptetaphimass
from gabbro.utils.multiclass import remove_class_tokens
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()


class GenEvalCallback(L.Callback):
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
        # Load class token dict, if it exists
        self.class_token_dict = trainer.datamodule.train_dataset.token_id_cfg.get(
            "class_token_dict", None
        )
        if self.class_token is not None:
            self.generate_jets(trainer, pl_module, stage="val")
            self.plot_real_vs_gen_jets_only(trainer, stage="val")
        else:
            self.plot_real_vs_gen_jets(trainer, pl_module, stage="val")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        # Load class token dict, if it exists
        self.class_token_dict = trainer.datamodule.test_dataset.token_id_cfg.get(
            "class_token_dict", None
        )
        if self.class_token is not None:
            self.generate_jets(trainer, pl_module, stage="test")
            self.plot_real_vs_gen_jets_only(trainer, stage="test")
        else:
            self.plot_real_vs_gen_jets(trainer, pl_module, stage="test")

    @rank_zero_only
    def plot_real_vs_gen_jets(self, trainer, pl_module, stage):
        plot_utils.set_mpl_style()

        if hasattr(pl_module, "head_gen"):  # new style in mulihead models
            if pl_module.head_gen is None:
                self.pylogger.warning(
                    "No head_gen found in the model. Skipping generation and plotting."
                )
                return None

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        if stage == "val":
            token_ids_list = pl_module.val_input_list
            token_masks_list = pl_module.val_mask_list
            jet_features_list = pl_module.val_input_list_jet
        elif stage == "test":
            token_ids_list = pl_module.test_input_list
            token_masks_list = pl_module.test_mask_list
            jet_features_list = pl_module.test_input_list_jet
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # If the collate function has been used, different batches will have
        # been padded to different lengths and we can't use np.concatenate.
        if stage == "val":
            collate = getattr(trainer.datamodule.val_dataset, "collate", False)
        elif stage == "test":
            collate = getattr(trainer.datamodule.test_dataset, "collate", False)
        if not collate:
            np_real_token_ids = np.concatenate(token_ids_list)
            np_real_token_masks = np.concatenate(token_masks_list)
            self.pylogger.info(
                f"np_real_token_ids.shape: {np_real_token_ids.shape}"
            )  # (N_jets, len_jets, 2)
            self.pylogger.info(
                f"np_real_token_masks.shape: {np_real_token_masks.shape}"
            )  # (N_jets, len_jets)

            names = ["part_token_id"]
            if hasattr(pl_module.backbone, "n_token_groups"):
                if pl_module.backbone.n_token_groups > 1:
                    names = [
                        f"part_token_id_group_{i}"
                        for i in range(pl_module.backbone.n_token_groups)
                    ]
            real_token_ids = np_to_ak(
                x=np_real_token_ids[
                    :, :, 0::2
                ],  # Slicing is needed in case multidim tokens are being used
                names=names,
                mask=np_real_token_masks,
            )
        else:
            # TODO: Implement collate also for multidim tokens. Currently this
            # has not been tested, which is why this section lacks multidim logic.
            names = ["part_token_id"]
            list_real_token_ids = []
            for i, batch in enumerate(token_ids_list):
                token_ids = np_to_ak(
                    x=batch[:, :, 0::2],
                    names=names,
                    mask=token_masks_list[i],
                )
                list_real_token_ids.append(token_ids)
            real_token_ids = ak.concatenate(list_real_token_ids)

        self.real_token_ids = ak.values_astype(real_token_ids[names], "int64")
        # if its epoch 0 step 0, just generate 100 jets
        n_gen_jets = self.n_val_gen_jets
        if stage == "val" and trainer.current_epoch == 0 and trainer.global_step == 0:
            n_gen_jets = 10

        # check if jet features are used, and if yes get the jet features as input
        # for generation
        if hasattr(pl_module.backbone, "jet_features_input_dim"):
            if pl_module.backbone.jet_features_input_dim > 0:
                self.pylogger.info("Using jet features as input for generation.")
                jet_features = torch.tensor(np.concatenate(jet_features_list)).to(pl_module.device)
            else:
                jet_features = None
        else:
            jet_features = None

        self.generation_output = pl_module.generate_n_jets_batched(
            n_jets=n_gen_jets,
            batch_size=self.batch_size_for_generation,
            seed=self.seed,
            x_jet=jet_features,
        )
        if pl_module.hparams.use_continuous_input:
            self.gen_token_ids = self.generation_output["part_token_id"]
            # field_names_continuous = [
            #     field for field in self.generation_output.fields if field != "part_token_id"
            # ]
            # self.gen_continuous_features = self.generation_output[field_names_continuous]
        else:
            # this already gives a {part_token_id: ...} ak.Array which will be
            # nested again when it is saved to parquet, resulting in a nested
            # {part_token_id: {part_token_id: ...}} structure
            # (this was the easiest to solve the difference between single-dim
            # and multi-dim tokens at some point)
            self.gen_token_ids = (
                ak.Array({"part_token_id": self.generation_output})
                if len(self.generation_output.fields) == 0
                else self.generation_output
            )

        self.pylogger.info(f"real_token_ids: {self.real_token_ids}")
        self.pylogger.info(f"gen_token_ids: {self.gen_token_ids}")

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)
        if stage == "val":
            filename_real = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_jets_token_ids.parquet"
        elif stage == "test":
            filename_real = f"{plot_dir}/test_real_jets_token_ids.parquet"
        else:
            raise ValueError(f"Unknown stage: {stage}")
        filename_gen = filename_real.replace("real_jets", "gen_jets")

        ak.to_parquet(ak.Array({"part_token_id": self.real_token_ids}), filename_real)
        ak.to_parquet(ak.Array({"part_token_id": self.gen_token_ids}), filename_gen)
        self.pylogger.info(f"Real jets saved to {filename_real}")
        self.pylogger.info(f"Generated jets saved to {filename_gen}")

        if pl_module.hparams.use_continuous_input:
            self.particles_real = pl_module.convert_valtest_batches_to_ak(stage)
            self.particles_gen = self.generation_output
            self.pylogger.info("Calculating the p4s from the pt, eta, phi, mass.")
            p4s_gen = p4s_from_ptetaphimass(self.particles_gen)
            p4s_real = p4s_from_ptetaphimass(self.particles_real)
        else:
            # reconstruct the real token_ids
            token_dir = Path(pl_module.token_dir)
            config_path = token_dir / "config.yaml"
            # load config to check if it's a binning or vqvae model
            cfg = OmegaConf.load(config_path)
            if "model" in cfg:
                tokenizer_type = "vqvae"
            else:
                tokenizer_type = "binning"
            common_reco_kwargs = dict(
                config_path=config_path,
                model_ckpt_path=token_dir / "model_ckpt.ckpt",
                device=pl_module.device,
                start_token_included=True,
                shift_tokens_by_minus_one=True,
                tokenizer_type=tokenizer_type,
                pad_length=cfg.data.dataset_kwargs_common.pad_length,
            )
            p4s_real, self.particles_real = reconstruct_jetclass_file(
                filename_in=filename_real,
                # those are the tokens *without* the stop token (the input to the model)
                end_token_included=False,
                **common_reco_kwargs,
            )
            p4s_gen, self.particles_gen = reconstruct_jetclass_file(
                filename_in=filename_gen,
                end_token_included=False,  # generated sequences don't have the stop token
                **common_reco_kwargs,
            )

            # save the p4s
            self.real_reco_p4s_filename = filename_real.replace(
                "_token_ids.parquet", "_reco_p4s.parquet"
            )
            self.gen_reco_p4s_filename = filename_gen.replace(
                "_token_ids.parquet", "_reco_p4s.parquet"
            )
            ak.to_parquet(p4s_real, self.real_reco_p4s_filename)
            ak.to_parquet(p4s_gen, self.gen_reco_p4s_filename)
            # save the x_ak arrays
            self.real_reco_x_filename = filename_real.replace(
                "_token_ids.parquet", "_reco_x.parquet"
            )
            self.gen_reco_x_filename = filename_gen.replace(
                "_token_ids.parquet", "_reco_x.parquet"
            )
            ak.to_parquet(self.particles_real, self.real_reco_x_filename)
            ak.to_parquet(self.particles_gen, self.gen_reco_x_filename)

            p4s_real = ak.from_parquet(self.real_reco_p4s_filename)
            p4s_gen = ak.from_parquet(self.gen_reco_p4s_filename)
            self.particles_real = ak.from_parquet(self.real_reco_x_filename)
            self.particles_gen = ak.from_parquet(self.gen_reco_x_filename)

        self.pylogger.info("Calculating the jet substructure.")
        substructure_real = JetSubstructure(p4s_real[ak.num(p4s_real) >= 3])
        substructure_gen = JetSubstructure(p4s_gen[ak.num(p4s_gen) >= 3])

        self.pylogger.info("Plotting the jet substructure.")
        substructure_real_ak = substructure_real.get_substructure_as_ak_array()
        substructure_gen_ak = substructure_gen.get_substructure_as_ak_array()

        # --- plot jet-level features ---
        print(f"Plotting {len(p4s_real)} real jets and {len(p4s_gen)} generated jets...")

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

        if self.bins_dict != {}:
            if isinstance(bins_dict, dict):
                common_keys = [key for key in self.bins_dict if key in bins_dict]
                for key in common_keys:
                    bins_dict[key] = self.bins_dict[key]
            else:
                self.pylogger.warning(
                    f"The provided bins_dict is not a dictionary, but {type(bins_dict)}. Falling back on default dict."
                )

        fig, axarr = plot_features(
            ak_array_dict={
                "Real jets (tokenized+reco)": substructure_real_ak,
                "Gen. jets": substructure_gen_ak,
            },
            names=names_labels_dict_for_plotting,
            bins_dict=bins_dict,
            flatten=False,
            ax_rows=2,
            legend_only_on=0,
            legend_kwargs={"loc": "upper left"},
            ax_size=(3, 2),
            ratio=True,
        )

        # for each particle and jet feature, calculate the mean and std of the
        # feature for both real and generated jets and store in a dict
        feature_stats_dict = {}
        for name in names_labels_dict_for_plotting.keys():
            feature_stats_dict[name] = {
                "mean_real": np.mean(substructure_real_ak[name]),
                "std_real": np.std(substructure_real_ak[name]),
                "mean_gen": np.mean(substructure_gen_ak[name]),
                "std_gen": np.std(substructure_gen_ak[name]),
            }
        for field_name in self.particles_gen.fields:
            feature_stats_dict[field_name] = {
                "mean_real": np.mean(self.particles_real[field_name]),
                "std_real": np.std(self.particles_real[field_name]),
                "mean_gen": np.mean(self.particles_gen[field_name]),
                "std_gen": np.std(self.particles_gen[field_name]),
            }
        self.pylogger.info("Feature stats dict:")
        for key, value in feature_stats_dict.items():
            self.pylogger.info(f"{key}: {value}")

        # log the average number of constituents
        if self.comet_logger is not None:
            self.comet_logger.log_metric(
                "val_avg_n_constituents_generated",
                np.mean(substructure_gen_ak["jet_n_constituents"]),
                step=trainer.global_step,
            )
            # log the difference in mean and std between real and gen for each feature
            for key, value in feature_stats_dict.items():
                self.comet_logger.log_metric(
                    f"val_mean_diff_{key}",
                    value["mean_gen"] - value["mean_real"],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"val_std_diff_{key}",
                    value["std_gen"] - value["std_real"],
                    step=trainer.global_step,
                )

        # calculate the kld between the real and generated jets
        kld_dict = calc_quantiled_kl_divergence_for_dict(
            dict_reference=substructure_real_ak,
            dict_approx=substructure_gen_ak,
            names=list(names_labels_dict_for_plotting.keys()),
            n_bins=50,
            return_zero_if_nan_or_inf=True,
        )
        self.pylogger.info(f"KLD values: {kld_dict}")

        if stage == "val":
            image_filename_jet_features = (
                f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_"
                f"jet_features_real_vs_gen_jets.{self.image_filetype}"
            )
        elif stage == "test":
            image_filename_jet_features = (
                f"{plot_dir}/test_jet_features_real_vs_gen_jets.{self.image_filetype}"
            )
        fig.savefig(image_filename_jet_features)
        self.pylogger.info(f"Saved jet features plot to {image_filename_jet_features}")

        # --- plot particle-level features ---
        # put `part_token_id` at the end because it leads to bad plot formatting
        # with the legend (this is just aesthetics)
        names_to_plot = [field for field in self.particles_real.fields if field != "part_token_id"]
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
        # TODO: also add that the particle token id is plotted in the case of non-continuous
        # input (currently only works for continuous input)
        if "part_token_id" in self.particles_real.fields:
            names_to_plot.append("part_token_id")
            bins_dict_for_particles["part_token_id"] = np.linspace(
                0, ak.max(self.particles_real["part_token_id"]), 101
            )

        fig, axarr = plot_features(
            ak_array_dict={
                "Real particles (tokenized+reco)": ak.Array(
                    {name: self.particles_real[name] for name in names_to_plot}
                ),
                "Gen. particles": ak.Array(
                    {name: self.particles_gen[name] for name in names_to_plot}
                ),
            },
            names={name: DEFAULT_LABELS.get(name, name) for name in names_to_plot},
            legend_only_on=0,
            legend_kwargs={"loc": "upper left"},
            ax_rows=math.ceil(len(self.particles_real.fields) / 4),
            ax_size=(3, 2),
            ratio=True,
            bins_dict=bins_dict_for_particles,
            logscale_features=["part_pt"],
            decorate_ax_kwargs={"yscale": 1.5},
        )
        image_filename_particle_features = image_filename_jet_features.replace(
            "jet_features", "particle_features"
        )
        fig.savefig(image_filename_particle_features)
        self.pylogger.info(f"Saved particle features plot to {image_filename_particle_features}")

        plt.show()

        if self.comet_logger is not None:
            for key, value in kld_dict.items():
                self.comet_logger.log_metric(f"val_kld_{key}", value, step=trainer.global_step)
            for filename in [
                image_filename_jet_features,
                image_filename_particle_features,
            ]:
                self.comet_logger.log_image(
                    filename,
                    name=filename.split("/")[-1],
                    step=trainer.global_step,
                )

        plt.close(fig)
        del fig, axarr
        gc.collect()

    @rank_zero_only
    def generate_jets(self, trainer, pl_module, stage):
        # TODO: evaluate how long it takes to generate, and decide if we want
        # to open this function for all processes in the case of final generation.
        # In that case, need to implement saving and loading with rank added
        # to the filename.
        # Get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        # Convert the numpy arrays and masks of the real jets to ak arrays of token IDs
        # Regarding pl_module.val_input_list etc: if you set eg. limit_val_batches=20, you'll
        # have 20 validation batches and thus only 20 * batch_size jets in pl_module.val_input_list
        if stage == "val":
            token_ids_list = pl_module.val_input_list
            token_masks_list = pl_module.val_mask_list
            self.class_token_dict = trainer.datamodule.train_dataset.token_id_cfg.get(
                "class_token_dict", None
            )
        elif stage == "test":
            token_ids_list = pl_module.test_input_list
            token_masks_list = pl_module.test_mask_list
            self.class_token_dict = trainer.datamodule.test_dataset.token_id_cfg.get(
                "class_token_dict", None
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if self.class_token is not None:
            assert self.class_token_dict is not None, (
                "Can't generate specific class when not trained with class_token_dict."
            )
            assert self.class_token in self.class_token_dict.values(), (
                f"Class token {self.class_token} not present in class_token_dict {self.class_token_dict.values()}"
            )

        # If the collate function has been used, different batches will have
        # been padded to different lengths and we can't use np.concatenate.
        if stage == "val":
            collate = getattr(trainer.datamodule.val_dataset, "collate", False)
        elif stage == "test":
            collate = getattr(trainer.datamodule.test_dataset, "collate", False)
        if not collate:
            np_real_token_ids = np.concatenate(token_ids_list)
            np_real_token_masks = np.concatenate(token_masks_list)
            self.pylogger.info(
                f"np_real_token_ids.shape: {np_real_token_ids.shape}"
            )  # (N_jets, len_jets, 2)
            self.pylogger.info(
                f"np_real_token_masks.shape: {np_real_token_masks.shape}"
            )  # (N_jets, len_jets)

            names = ["part_token_id"]

            if hasattr(pl_module.module, "n_token_groups"):
                if pl_module.module.n_token_groups > 1:
                    names = [
                        f"part_token_id_group_{i}" for i in range(pl_module.module.n_token_groups)
                    ]
            else:
                if (
                    self.class_token is not None
                ):  # TODO: Implement class token also for multidim tokens
                    # Pick only the jets with the correct class token (class token was attached via the dataloader)
                    # Get the input (as opposed to targets) using [:, :, 0], then require the class token to be correct
                    selected_jets = np.where(
                        (np_real_token_ids[:, :, 0])[:, 1] == self.class_token
                    )
                    np_real_token_ids = np.squeeze(np_real_token_ids[selected_jets, :, :])
                    np_real_token_masks = np.squeeze(np_real_token_masks[selected_jets, :])
                    self.pylogger.info(
                        f"np_real_token_ids.shape after selecting only jets with class token {self.class_token}: {np_real_token_ids.shape}"
                    )  # (N_jets, len_jets, 2)
                    self.pylogger.info(
                        f"np_real_token_masks.shape after selecting only jets with class token {self.class_token}: {np_real_token_masks.shape}"
                    )  # (N_jets, len_jets)

            real_token_ids = np_to_ak(
                x=np_real_token_ids[
                    :, :, 0::2
                ],  # Slicing is needed in case multidim tokens are being used
                names=names,
                mask=np_real_token_masks,
            )
        else:
            # TODO: Implement collate also for multidim tokens. Currently this
            # has not been tested, which is why this section lacks multidim logic.
            # TODO: The collate function currently doesn't work for multiclass
            names = ["part_token_id"]
            list_real_token_ids = []
            for i, batch in enumerate(token_ids_list):
                token_ids = np_to_ak(
                    x=batch[:, :, 0::2],
                    names=names,
                    mask=token_masks_list[i],
                )
                list_real_token_ids.append(token_ids)
            real_token_ids = ak.concatenate(list_real_token_ids)

        self.real_token_ids = ak.values_astype(real_token_ids["part_token_id"], "int64")

        # Generate
        if stage == "val":
            n_jets = self.n_val_gen_jets
        elif stage == "test":
            n_jets = self.n_final_gen_jets

        if self.class_token_dict is not None:
            if not collate:
                # TODO: The collate function currently doesn't work for multiclass
                n_classes = len(self.class_token_dict)
                if self.class_token is not None:
                    self.pylogger.info(f"Generating jets with class token {self.class_token}")
                    self.gen_token_ids = pl_module.generate_n_jets_batched_multiclass(
                        n_jets,
                        batch_size=self.batch_size_for_generation,
                        class_token=self.class_token,  # Generate a specific class
                        n_classes=n_classes,
                    )
                else:
                    self.gen_token_ids = pl_module.generate_n_jets_batched_multiclass(
                        n_jets,
                        batch_size=self.batch_size_for_generation,
                        class_token=None,  # None: generate from all classes at will
                        n_classes=n_classes,
                    )
        else:
            self.gen_token_ids = pl_module.generate_n_jets_batched_multiclass(
                n_jets,
                batch_size=self.batch_size_for_generation,
                class_token=None,
                n_classes=None,  # None: indicates that model did not train with class_token_dict
            )
        self.pylogger.info(f"real_token_ids: {self.real_token_ids}")
        self.pylogger.info(f"gen_token_ids: {self.gen_token_ids}")
        self.pylogger.info(f"Length of generated jets: {len(self.gen_token_ids)}")
        self.pylogger.info(f"Length of real jets: {len(self.real_token_ids)}")

        # log min max values of the token IDs and of the number of constituents
        multiplicity_real = ak.num(self.real_token_ids)
        multiplicity_gen = ak.num(self.gen_token_ids)
        self.pylogger.info(
            f"Real jets: min multiplicity: {ak.min(multiplicity_real)}, "
            f"max multiplicity: {ak.max(multiplicity_real)}"
        )
        self.pylogger.info(
            f"Gen jets: min multiplicity: {ak.min(multiplicity_gen)}, "
            f"max multiplicity: {ak.max(multiplicity_gen)}"
        )
        self.pylogger.info(
            f"Real jets: min token id: {ak.min(self.real_token_ids)}, "
            f"max token id: {ak.max(self.real_token_ids)}"
        )
        self.pylogger.info(
            f"Gen jets: min token id: {ak.min(self.gen_token_ids)}, "
            f"max token id: {ak.max(self.gen_token_ids)}"
        )

        # Check if there are nan values in the token IDs
        if np.sum(np.isnan(ak.flatten(self.real_token_ids))) > 0:
            self.pylogger.warning("Real token ids contain NaN values.")
        if np.sum(np.isnan(ak.flatten(self.gen_token_ids))) > 0:
            self.pylogger.warning("Generated token ids contain NaN values.")

        # Save jets (in plot_dir, since they will be plotted afterwards)
        self.plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(self.plot_dir, exist_ok=True)

        if stage == "val":
            filename_real = f"{self.plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_jets_token_ids.parquet"
        elif stage == "test":
            filename_real = f"{self.plot_dir}/test_real_jets_token_ids.parquet"
        else:
            raise ValueError(f"Unknown stage: {stage}")
        if self.class_token_dict is not None:
            if self.class_token is None:
                filename_real = filename_real.replace("token_ids", "token_ids_all_classes")
            if self.class_token is not None:
                # Extract the corresponding (string) label for the class token
                for label, token in self.class_token_dict.items():
                    if token == self.class_token:
                        class_label = label
                filename_real = filename_real.replace("token_ids", f"token_ids_{class_label}")
        filename_gen = filename_real.replace("real_jets", "gen_jets")

        ak.to_parquet(self.real_token_ids, filename_real)
        ak.to_parquet(self.gen_token_ids, filename_gen)
        # ak.to_parquet(ak.Array({"part_token_id": self.real_token_ids}), filename_real)
        # ak.to_parquet(ak.Array({"part_token_id": self.gen_token_ids}), filename_gen)

        self.pylogger.info(f"Real jets saved to {filename_real}")
        self.pylogger.info(f"Generated jets saved to {filename_gen}")

        self.file_real_token_ids_with_class_label = filename_real
        self.file_gen_token_ids_with_class_label = filename_gen

        # Remove class tokens and save
        if self.class_token_dict is not None:
            # From real jets
            self.real_token_ids = remove_class_tokens(
                self.real_token_ids, len(self.class_token_dict)
            )
            filename_real = filename_real.replace("real_jets", "real_jets_no_classtoken")
            ak.to_parquet(self.real_token_ids, filename_real)
            self.pylogger.info(f"Real jets saved without class tokens to {filename_real}")
            self.pylogger.info(
                f"Real jets after removing class token: min token id: {ak.min(self.real_token_ids)}, "
                f"max token id: {ak.max(self.real_token_ids)}"
            )
            # From generated jets
            self.gen_token_ids = remove_class_tokens(
                self.gen_token_ids, len(self.class_token_dict)
            )
            filename_gen = filename_gen.replace("gen_jets", "gen_jets_no_classtoken")
            ak.to_parquet(self.gen_token_ids, filename_gen)
            self.pylogger.info(f"Generated jets saved without class tokens to {filename_gen}")
            self.pylogger.info(
                f"Gen jets after removing class token: min token id: {ak.min(self.gen_token_ids)}, "
                f"max token id: {ak.max(self.gen_token_ids)}"
            )

        # reconstruct the physics from the token_ids
        token_dir = Path(pl_module.token_dir)
        common_reco_kwargs = dict(
            config_path=token_dir / "config.yaml",
            model_ckpt_path=token_dir / "model_ckpt.ckpt",
            device=pl_module.device,
            start_token_included=True,
            shift_tokens_by_minus_one=True,
        )
        p4s_real, x_ak_real = reconstruct_jetclass_file(
            filename_in=filename_real,
            end_token_included=True,
            **common_reco_kwargs,
        )
        p4s_gen, x_ak_gen = reconstruct_jetclass_file(
            filename_in=filename_gen,
            end_token_included=False,
            **common_reco_kwargs,
        )

        # TODO: add plotting of all non-p4 features if there are any (or particle-level
        # features in general)

        # save the p4s
        self.real_reco_p4s_filename = filename_real.replace("_token_ids", "_reco_p4s")
        self.gen_reco_p4s_filename = filename_gen.replace("_token_ids", "_reco_p4s")
        ak.to_parquet(p4s_real, self.real_reco_p4s_filename)
        ak.to_parquet(p4s_gen, self.gen_reco_p4s_filename)
        self.pylogger.info(
            f"Saved p4s to {self.real_reco_p4s_filename} (real) and {self.gen_reco_p4s_filename} (gen)"
        )

        # save the x_ak arrays
        self.real_reco_x_filename = filename_real.replace("_token_ids", "_reco_x")
        self.gen_reco_x_filename = filename_gen.replace("_token_ids", "_reco_x")
        ak.to_parquet(x_ak_real, self.real_reco_x_filename)
        ak.to_parquet(x_ak_gen, self.gen_reco_x_filename)

        p4s_real = ak.from_parquet(self.real_reco_p4s_filename)
        p4s_gen = ak.from_parquet(self.gen_reco_p4s_filename)
        x_ak_real = ak.from_parquet(self.real_reco_x_filename)
        x_ak_gen = ak.from_parquet(self.gen_reco_x_filename)

        self.pylogger.info("Calculating the jet substructure.")
        self.pylogger.info(f"Real tokens from file {self.real_reco_p4s_filename}")
        self.pylogger.info(f"Gen tokens from file {self.gen_reco_p4s_filename}")
        substructure_real = JetSubstructure(p4s_real[ak.num(p4s_real) >= 3])
        substructure_gen = JetSubstructure(p4s_gen[ak.num(p4s_gen) >= 3])

        self.substructure_real_ak = substructure_real.get_substructure_as_ak_array()
        self.substructure_gen_ak = substructure_gen.get_substructure_as_ak_array()

        # Calculate the KLD between the real and generated jets
        self.names_labels_dict_for_kld = {
            "jet_pt": "Jet $p_T$ [GeV]",
            "jet_eta": "Jet $\\eta$",
            "jet_phi": "Jet $\\phi$",
            "jet_mass": "Jet mass [GeV]",
            "tau32": "$\\tau_{32}$",
            "tau21": "$\\tau_{21}$",
            "jet_n_constituents": "Number of constituents",
        }

        self.kld_dict = calc_quantiled_kl_divergence_for_dict(
            dict_reference=self.substructure_real_ak,
            dict_approx=self.substructure_gen_ak,
            names=list(self.names_labels_dict_for_kld.keys()),
            n_bins=50,
            return_zero_if_nan_or_inf=True,
        )
        self.pylogger.info(f"KLD values: {self.kld_dict}")

        if self.comet_logger is not None:
            for key, value in self.kld_dict.items():
                self.comet_logger.log_metric(f"val_kld_{key}", value, step=trainer.global_step)

    @rank_zero_only
    def plot_real_vs_gen_jets_only(self, trainer, stage):
        # TODO: add plotting of all non-p4 features if there are any (or particle-level
        # features in general)
        # TODO: Here it should preferably not only be able to plot one class,
        # but to loop over several classes and plot them together.
        # TODO: Plot class-wise even when we plot all classes, so we can see if
        # all classes are generated equally well.
        # Get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment
        # Load jets to be plotted
        p4s_real = ak.from_parquet(self.real_reco_p4s_filename)
        p4s_gen = ak.from_parquet(self.gen_reco_p4s_filename)
        x_ak_real = ak.from_parquet(self.real_reco_x_filename)
        x_ak_gen = ak.from_parquet(self.gen_reco_x_filename)
        # --- plot jet-level features ---
        print(f"Plotting {len(p4s_real)} real jets and {len(p4s_gen)} generated jets...")
        fig, _ = plot_features(
            ak_array_dict={
                "Real jets (tokenized+reco)": self.substructure_real_ak,
                "Gen. jets": self.substructure_gen_ak,
            },
            names=self.names_labels_dict_for_kld,
            bins_dict={
                "jet_pt": np.linspace(0, 1050, 100),
                "jet_eta": np.linspace(-0.1, 0.1, 70),
                "jet_phi": np.linspace(-0.01, 0.01, 70),
                "jet_mass": np.linspace(0, 250, 70),
                "tau32": np.linspace(0, 1.2, 70),
                "tau21": np.linspace(0, 1.2, 70),
                "jet_n_constituents": np.linspace(-0.5, 128.5, 130),
            },
            flatten=False,
            ax_rows=2,
            legend_only_on=0,
        )
        if stage == "val":
            image_filename_jet_features = (
                f"{self.plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_"
                f"val_jet_features_real_vs_gen_jets.{self.image_filetype}"
            )
        elif stage == "test":
            image_filename_jet_features = (
                f"{self.plot_dir}/test_jet_features_real_vs_gen_jets.{self.image_filetype}"
            )
        if self.class_token_dict is not None:
            if self.class_token is None:
                image_filename_jet_features = image_filename_jet_features.replace(
                    "_jets", "_jets_all_classes"
                )
            if self.class_token is not None:
                # Extract the corresponding (string) label for the class token
                for label, token in self.class_token_dict.items():
                    if token == self.class_token:
                        class_label = label
                image_filename_jet_features = image_filename_jet_features.replace(
                    "_jets", f"_jets_{class_label}"
                )
        fig.savefig(image_filename_jet_features)
        self.pylogger.info(f"Saved jet features plot to {image_filename_jet_features}")
        # --- plot particle-level features ---
        fig, _ = plot_features(
            ak_array_dict={
                "Real particles": ak.Array({name: x_ak_real[name] for name in x_ak_real.fields}),
                "Gen. particles": ak.Array({name: x_ak_gen[name] for name in x_ak_gen.fields}),
            },
            names=x_ak_real.fields,
            legend_only_on=0,
            ax_rows=math.ceil(len(x_ak_real.fields) / 4),
        )
        image_filename_particle_features = image_filename_jet_features.replace(
            "jet_features", "particle_features"
        )
        fig.savefig(image_filename_particle_features)
        self.pylogger.info(f"Saved particle features plot to {image_filename_particle_features}")
        plt.show()
        if self.comet_logger is not None:
            for key, value in self.kld_dict.items():
                self.comet_logger.log_metric(f"val_kld_{key}", value, step=trainer.global_step)
            for filename in [
                image_filename_jet_features,
                image_filename_particle_features,
            ]:
                self.comet_logger.log_image(
                    filename, name=filename.split("/")[-1], step=trainer.global_step
                )

        plt.close(fig)
        del fig
        gc.collect()
