from typing import Any, Optional

import awkward as ak
import numpy as np
import torch
import vector
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, ak_to_np_stack
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)


class TauDataset(Dataset):
    def __init__(
        self,
        # filename,
        # n_jets,
        feature_dict,
        pad_length,
        **kwargs,
    ):
        logger.info("Loading data from parquet file")
        ak_part_features, labels_train = load_tau_jets_from_parquet(
            # filename,
            feature_dict=feature_dict,
            # n_jets=n_jets,
        )

        ak_x_padded, ak_mask = ak_pad(
            ak_part_features,
            maxlen=pad_length,
            fill_value=0.0,
            return_mask=True,
        )
        self.part_features = torch.tensor(ak_to_np_stack(ak_x_padded, names=feature_dict.keys()))
        self.part_mask = torch.tensor(ak.to_numpy(ak_mask))
        self.jet_type_labels = torch.tensor(ak.to_numpy(labels_train))

        # Print shapes
        logger.info(f"part_features.shape: {self.part_features.shape}")
        logger.info(f"part_mask.shape: {self.part_mask.shape}")
        logger.info(f"jet_type_labels.shape: {self.jet_type_labels.shape}")

    def __getitem__(self, index):
        return {
            "part_features": self.part_features[index],
            "part_mask": self.part_mask[index],
            "jet_type_labels": self.jet_type_labels[index],
        }

    def __len__(self):
        return len(self.part_features)


def to_p4(p4_obj):
    """Helper function to convert awkward array to 4-momentum vector."""
    return vector.awk(
        ak.zip(
            {
                "mass": p4_obj.tau,
                "x": p4_obj.x,
                "y": p4_obj.y,
                "z": p4_obj.z,
            }
        )
    )


def load_tau_jets_from_parquet(
    # parquet_filename,
    feature_dict: dict,
    # n_jets: int = None,
    stage="train",
    shuffle_seed=2,
):
    """Helper function to load tau jets from parquet file.

    Parameters
    ----------
    parquet_filename : str
        Path to parquet file
    feature_features : list
        Dictionary of features to load from the parquet file and their preprocessing
        parameters
    n_jets : int
        Number of jets to load from the parquet file
    stage : str
        Stage of the dataset (train, val, test)
    shuffle_seed : int
        Seed for shuffling the dataset

    Returns
    -------
    ak.Array
        ak.Array of jet constituent features
    ak.Array
        ak.Array of jet labels (0 for QCD, 1 for top)
    """

    split_indices = {
        # "zh": (10_000, 20_000),  # 530_722 in total
        # "z": (10_000, 20_000),  # 457_547 in total
        # "qq": (10_000, 20_000),  # 3_523_428 in total
        "zh": (350_000, 400_000),  # 530_722 in total
        "z": (350_000, 400_000),  # 457_547 in total
        "qq": (350_000, 400_000),  # 3_523_428 in total
    }

    dataset_path = "/beegfs/desy/user/birkjosc/datasets/ML-tau-data"

    start_idx = {}
    end_idx = {}

    logger.info(f"Loading data for stage: {stage}")

    if stage == "train":
        start_idx["zh"] = 0
        end_idx["zh"] = split_indices["zh"][0]
        start_idx["z"] = 0
        end_idx["z"] = split_indices["z"][0]
        start_idx["qq"] = 0
        end_idx["qq"] = split_indices["qq"][0]
    elif stage == "val":
        start_idx["zh"] = split_indices["zh"][0]
        end_idx["zh"] = split_indices["zh"][1]
        start_idx["z"] = split_indices["z"][0]
        end_idx["z"] = split_indices["z"][1]
        start_idx["qq"] = split_indices["qq"][0]
        end_idx["qq"] = split_indices["qq"][1]
    elif stage == "test":
        start_idx["zh"] = split_indices["zh"][1]
        end_idx["zh"] = None
        start_idx["z"] = split_indices["z"][1]
        end_idx["z"] = None
        start_idx["qq"] = split_indices["qq"][1]
        end_idx["qq"] = 500_000

    qq_data = ak.from_parquet(f"{dataset_path}/qq.parquet")[start_idx["qq"] : end_idx["qq"]]
    z_data = ak.from_parquet(f"{dataset_path}/z.parquet")[start_idx["z"] : end_idx["z"]]
    zh_data = ak.from_parquet(f"{dataset_path}/zh.parquet")[start_idx["zh"] : end_idx["zh"]]
    ak_arr = ak.concatenate([qq_data, z_data, zh_data])
    labels = ak.concatenate(
        [
            np.zeros(len(qq_data)),
            np.ones(len(z_data)),
            2 * np.ones(len(zh_data)),
        ]
    )
    rng = np.random.default_rng(shuffle_seed)
    idx = rng.permutation(len(ak_arr))
    ak_arr = ak_arr[idx]
    labels = labels[idx]

    batch_size = 100_000

    n_jets = len(ak_arr)

    n_batches = (n_jets + batch_size - 1) // batch_size

    all_particle_features_list = []
    labels_list = []

    # Loop over batches and calculate all possible features
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_jets)
        batch_ak_arr = ak_arr[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        batch_p4 = to_p4(batch_ak_arr.reco_cand_p4s)

        jets = ak.sum(batch_p4, axis=1)

        batch_particle_features = ak.Array(
            {
                "part_pt": batch_p4.pt,
                "part_eta": batch_p4.eta,
                "part_phi": batch_p4.phi,
                "part_energy": batch_p4.energy,
                "part_etarel": batch_p4.deltaeta(jets),
                "part_phirel": batch_p4.deltaphi(jets),
                "part_ptrel": batch_p4.pt / jets.pt,
            }
        )
        all_particle_features_list.append(batch_particle_features)
        labels_list.append(batch_labels)

    all_particle_features = ak.concatenate(all_particle_features_list)
    labels = ak.concatenate(labels_list)

    all_particle_features_selected_and_pp = ak_select_and_preprocess(
        all_particle_features, pp_dict=feature_dict
    )

    # filter out jets with 0 particles
    mask = ak.num(getattr(all_particle_features_selected_and_pp, list(feature_dict.keys())[0])) > 0
    all_particle_features_selected_and_pp = all_particle_features_selected_and_pp[mask]
    labels = labels[mask]
    return all_particle_features_selected_and_pp, labels


class TauDataModule(LightningDataModule):
    """Data module for the tau dataset."""

    def __init__(
        self,
        # file_train: str = None,
        # file_val: str = None,
        # file_test: str = None,
        # n_jets_train: int = None,
        # n_jets_val: int = None,
        # n_jets_test: int = None,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()
        # if file_train is None or file_val is None or file_test is None:
        #     raise ValueError("file_train, file_val and file_test must be specified")

    def prepare_data(self) -> None:
        """Prepare the data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`, so be
        careful not to execute things like random split twice! Also, it is
        called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once the
        data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`,
        `"test"`, or `"predict"`. Defaults to ``None``.
        """

        if stage == "fit":
            # # Create datasets
            # logger.info(f"Loading train data: {self.hparams.file_train}")
            self.data_train = TauDataset(
                # filename=self.hparams.file_train,
                # n_jets=self.hparams.n_jets_train,
                feature_dict=OmegaConf.to_container(
                    self.hparams.dataset_kwargs_common.feature_dict
                ),
                pad_length=self.hparams.pad_length,
            )

            # logger.info(f"Loading val data: {self.hparams.file_val}")
            self.data_val = TauDataset(
                # filename=self.hparams.file_val,
                # n_jets=self.hparams.n_jets_val,
                feature_dict=OmegaConf.to_container(
                    self.hparams.dataset_kwargs_common.feature_dict
                ),
                pad_length=self.hparams.pad_length,
            )
        elif stage == "test":
            # logger.info(f"Loading test data: {self.hparams.file_test}")
            self.data_test = TauDataset(
                # filename=self.hparams.file_test,
                # n_jets=self.hparams.n_jets_test,
                feature_dict=OmegaConf.to_container(
                    self.hparams.dataset_kwargs_common.feature_dict
                ),
                pad_length=self.hparams.pad_length,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
