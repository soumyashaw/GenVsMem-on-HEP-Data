"""Data utilities for loading and preprocessing LHCO H5 data without tokenization."""

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gabbro.data.loading import (
    load_lhco_jets_from_h5,
    load_multiple_h5_files,
    load_case_jets_from_h5,
    load_multiple_case_h5_files,
)
from gabbro.utils.arrays import ak_pad, ak_to_np_stack


def load_lhco_continuous_data_from_h5(
    signal_path: str,
    background_path: str,
    n_signal: int,
    n_background: int,
    feature_dict: dict,
    max_sequence_len: int = 128,
    jet_name: str = "jet1",
    mom4_format: str = "epxpypz",
):
    """Load LHCO data from H5 files as continuous features (no tokenization).
    
    Parameters
    ----------
    signal_path : str
        Path to signal H5 file
    background_path : str
        Path to background H5 file
    n_signal : int
        Number of signal jets to load
    n_background : int
        Number of background jets to load
    feature_dict : dict
        Feature dictionary with preprocessing parameters
    max_sequence_len : int
        Maximum sequence length for padding
    jet_name : str
        Name of jet group in H5 file ("jet1" or "jet2")
    mom4_format : str
        Format of 4-momentum in H5 file
        
    Returns
    -------
    dict
        Dictionary with keys:
        - part_features: (N, seq_len, n_features)
        - part_mask: (N, seq_len)
        - jet_type_labels: (N,)
    """
    # Load signal jets
    signal_features, signal_labels = load_lhco_jets_from_h5(
        h5_filename=signal_path,
        feature_dict=feature_dict,
        n_jets=n_signal,
        jet_name=jet_name,
        mom4_format=mom4_format,
    )
    
    # Load background jets
    background_features, background_labels = load_lhco_jets_from_h5(
        h5_filename=background_path,
        feature_dict=feature_dict,
        n_jets=n_background,
        jet_name=jet_name,
        mom4_format=mom4_format,
    )
    
    # Pad to max_sequence_len
    signal_padded, signal_mask = ak_pad(
        signal_features,
        maxlen=max_sequence_len,
        return_mask=True,
    )
    background_padded, background_mask = ak_pad(
        background_features,
        maxlen=max_sequence_len,
        return_mask=True,
    )
    
    # Convert to numpy
    feature_names = list(feature_dict.keys())
    signal_np = ak_to_np_stack(signal_padded, names=feature_names)
    background_np = ak_to_np_stack(background_padded, names=feature_names)
    
    # Concatenate signal and background
    part_features = np.concatenate([signal_np, background_np], axis=0)
    part_mask = np.concatenate([signal_mask, background_mask], axis=0)
    
    # Labels are now simple numpy arrays (0 or 1)
    jet_type_labels = np.concatenate([
        signal_labels,
        background_labels,
    ]).astype(np.int64)
    
    return {
        "part_features": part_features,
        "part_mask": part_mask,
        "jet_type_labels": jet_type_labels,
    }


def create_lhco_h5_dataloaders(
    h5_files_train,
    h5_files_val,
    feature_dict,
    batch_size=64,
    n_jets_train=None,
    n_jets_val=None,
    max_sequence_len=128,
    mom4_format="epxpypz",
    jet_name="jet1",
    train_val_split=None,
    shuffle_train=True,
    num_workers=1,
):
    """Create PyTorch DataLoaders from LHCO HDF5 files.
    
    Parameters
    ----------
    h5_files_train : list
        List of HDF5 file paths for training
    h5_files_val : list
        List of HDF5 file paths for validation
    feature_dict : dict
        Feature preprocessing dictionary
    batch_size : int
        Batch size for dataloaders
    n_jets_train : int or list of int, optional
        Number of jets to load for training.
        If int, total number split across files.
        If list, number per file (must match len(h5_files_train)).
        If None, load all jets.
    n_jets_val : int or list of int, optional
        Number of jets to load for validation (same behavior as n_jets_train)
    max_sequence_len : int
        Maximum sequence length (padding)
    mom4_format : str
        4-momentum format in HDF5 files
    jet_name : str
        Jet name in H5 files ("jet1" or "jet2")
    train_val_split : float, optional
        If provided (e.g., 0.8), will split the same data into train/val
        with this fraction for training. Ignores h5_files_val in this case.
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of workers for data loading
        
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """

    if jet_name in ["jet1", "jet2"]:
        if train_val_split is not None:
            # Case where the jet is either jet1 or jet2 and data is split into train/val
            # Load all data and split
            print(f"Loading data from H5 files and splitting {train_val_split:.1%}/{1-train_val_split:.1%} train/val...")
            
            all_features, all_labels = load_multiple_h5_files(
                h5_files_train,
                feature_dict,
                n_jets_per_file=n_jets_train,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )
            
            # Calculate split index
            n_total = len(all_features)
            n_train = int(n_total * train_val_split)

            print(f"Total jets loaded: {n_total}, Training jets: {n_train}, Validation jets: {n_total - n_train}")
            
            # Shuffle before splitting
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            print(f"Splitting {n_total} jets into {n_train} train / {n_total - n_train} val")
            
            # Split the data
            train_features = all_features[train_indices]
            val_features = all_features[val_indices]
            train_labels = all_labels[train_indices]
            val_labels = all_labels[val_indices]
            
        else:
            # Case where jet is either jet1 or jet2 and separate train/val files
            # Load training data
            train_features, train_labels = load_multiple_h5_files(
                h5_files_train,
                feature_dict,
                n_jets_per_file=n_jets_train,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )
            
            # Load validation data
            val_features, val_labels = load_multiple_h5_files(
                h5_files_val,
                feature_dict,
                n_jets_per_file=n_jets_val,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )
        
        # Pad and convert to numpy
        train_features_padded, train_mask = ak_pad(
            train_features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded, val_mask = ak_pad(
            val_features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack training features
        train_features_stacked = ak.concatenate(
            [train_features_padded[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features
        val_features_stacked = ak.concatenate(
            [val_features_padded[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        
        # Convert to numpy then to torch tensors
        train_x = torch.from_numpy(ak.to_numpy(train_features_stacked)).float()
        train_mask_t = torch.from_numpy(ak.to_numpy(train_mask)).float()
        # Labels are now just a simple numpy array (0 or 1)
        train_labels_t = torch.from_numpy(train_labels).long()
        
        val_x = torch.from_numpy(ak.to_numpy(val_features_stacked)).float()
        val_mask_t = torch.from_numpy(ak.to_numpy(val_mask)).float()
        # Labels are now just a simple numpy array (0 or 1)
        val_labels_t = torch.from_numpy(val_labels).long()
        
        print(f"Training data shape: {train_x.shape}, Labels: {train_labels_t.shape}")
        print(f"  - Signal: {train_labels_t.sum()}, Background: {(1-train_labels_t).sum()}")
        print(f"Validation data shape: {val_x.shape}, Labels: {val_labels_t.shape}")
        print(f"  - Signal: {val_labels_t.sum()}, Background: {(1-val_labels_t).sum()}")
    
    elif jet_name == "both":
        if train_val_split is not None:
            # Case where both jets are loaded and data is split into train/val
            # Load all data and split
            print(f"Loading data from H5 files and splitting {train_val_split:.1%}/{1-train_val_split:.1%} train/val...")
            
            features_jet1, features_jet2, all_labels = load_multiple_h5_files(
                h5_files_train,
                feature_dict,
                n_jets_per_file=n_jets_train,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )

            # Calculate split index
            n_total = len(features_jet1)
            n_train = int(n_total * train_val_split)

            print(f"Total jets loaded: {n_total}, Training jets: {n_train}, Validation jets: {n_total - n_train}")

            # Shuffle before splitting
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            print(f"Splitting {n_total} jets into {n_train} train / {n_total - n_train} val")
            
            # Split the data into train and val sets for both jets
            train_features_jet1 = features_jet1[train_indices]
            val_features_jet1 = features_jet1[val_indices]
            train_features_jet2 = features_jet2[train_indices]
            val_features_jet2 = features_jet2[val_indices]
            train_labels = all_labels[train_indices]
            val_labels = all_labels[val_indices]

        else:
            # Case where both jets are loaded and separate train/val files
            # Load training data
            train_features_jet1, train_features_jet2, train_labels = load_multiple_h5_files(
                h5_files_train,
                feature_dict,
                n_jets_per_file=n_jets_train,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )
            
            # Load validation data
            val_features_jet1, val_features_jet2, val_labels = load_multiple_h5_files(
                h5_files_val,
                feature_dict,
                n_jets_per_file=n_jets_val,
                mom4_format=mom4_format,
                jet_name=jet_name,
            )
        
        # Pad and convert to numpy for jet1
        train_features_padded_jet1, train_mask_jet1 = ak_pad(
            train_features_jet1, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded_jet1, val_mask_jet1 = ak_pad(
            val_features_jet1, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )

        # Pad and convert to numpy for jet2
        train_features_padded_jet2, train_mask_jet2 = ak_pad(
            train_features_jet2, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded_jet2, val_mask_jet2 = ak_pad(
            val_features_jet2, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )

        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack training features for jet1
        train_features_stacked_jet1 = ak.concatenate(
            [train_features_padded_jet1[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features for jet1
        val_features_stacked_jet1 = ak.concatenate(
            [val_features_padded_jet1[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )

        # Stack training features for jet2
        train_features_stacked_jet2 = ak.concatenate(
            [train_features_padded_jet2[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features for jet2
        val_features_stacked_jet2 = ak.concatenate(
            [val_features_padded_jet2[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )

        # Convert to numpy then to torch tensors
        train_x_jet1 = torch.from_numpy(ak.to_numpy(train_features_stacked_jet1)).float()
        train_mask_t_jet1 = torch.from_numpy(ak.to_numpy(train_mask_jet1)).float()
        # Labels are now just a simple numpy array (0 or 1)
        train_labels_t = torch.from_numpy(train_labels).long()
        
        val_x_jet1 = torch.from_numpy(ak.to_numpy(val_features_stacked_jet1)).float()
        val_mask_t_jet1 = torch.from_numpy(ak.to_numpy(val_mask_jet1)).float()
        # Labels are now just a simple numpy array (0 or 1)
        val_labels_t = torch.from_numpy(val_labels).long()

        train_x_jet2 = torch.from_numpy(ak.to_numpy(train_features_stacked_jet2)).float()
        train_mask_t_jet2 = torch.from_numpy(ak.to_numpy(train_mask_jet2)).float()
        
        val_x_jet2 = torch.from_numpy(ak.to_numpy(val_features_stacked_jet2)).float()
        val_mask_t_jet2 = torch.from_numpy(ak.to_numpy(val_mask_jet2)).float()

        print(f"Training data shape: {train_x_jet1.shape}, Labels: {train_labels_t.shape}")
        print(f"  - Signal: {train_labels_t.sum()}, Background: {(1-train_labels_t).sum()}")
        print(f"Validation data shape: {val_x_jet1.shape}, Labels: {val_labels_t.shape}")
        print(f"  - Signal: {val_labels_t.sum()}, Background: {(1-val_labels_t).sum()}")


    # Create custom dataset class
    class JetDataset(Dataset):
        def __init__(self, features, masks, labels, features_jet2=None, masks_jet2=None):
            self.features = features
            self.features_jet2 = features_jet2
            self.masks = masks
            self.masks_jet2 = masks_jet2
            self.labels = labels
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            result = {
                "part_features": self.features[idx],
                "part_mask": self.masks[idx],
                "jet_type_labels": self.labels[idx],
                "jet_features": torch.tensor([]),
            }
            # Only include jet2 fields if they exist (for dijet mode)
            if self.features_jet2 is not None:
                result["part_features_jet2"] = self.features_jet2[idx]
                result["part_mask_jet2"] = self.masks_jet2[idx]
            return result

    if jet_name in ["jet1", "jet2"]:
        # Create datasets
        train_dataset = JetDataset(features=train_x, masks=train_mask_t, labels=train_labels_t)
        val_dataset = JetDataset(features=val_x, masks=val_mask_t, labels=val_labels_t)

    elif jet_name == "both":
        # Create datasets with both jets
        train_dataset = JetDataset(features=train_x_jet1, masks=train_mask_t_jet1, 
                                   labels=train_labels_t, features_jet2=train_x_jet2, masks_jet2=train_mask_t_jet2)
        val_dataset = JetDataset(features=val_x_jet1, masks=val_mask_t_jet1,
                                 labels=val_labels_t, features_jet2=val_x_jet2, masks_jet2=val_mask_t_jet2)
    
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_custom_lhco_h5_dataloaders(
    h5_files_train,
    h5_files_val,
    feature_dict,
    batch_size=64,
    n_jets_train=None,
    n_jets_val=None,
    max_sequence_len=128,
    mom4_format="epxpypz",
    jet_name="jet1",
    train_val_split=None,
    shuffle_train=True,
    num_workers=1,
    **kwargs,
):
    """Create PyTorch DataLoaders from LHCO HDF5 files.
    
    Parameters
    ----------
    h5_files_train : list
        List of HDF5 file paths for training
    h5_files_val : list
        List of HDF5 file paths for validation
    feature_dict : dict
        Feature preprocessing dictionary
    batch_size : int
        Batch size for dataloaders
    n_jets_train : int or list of int, optional
        Number of jets to load for training.
        If int, total number split across files.
        If list, number per file (must match len(h5_files_train)).
        If None, load all jets.
    n_jets_val : int or list of int, optional
        Number of jets to load for validation (same behavior as n_jets_train)
    max_sequence_len : int
        Maximum sequence length (padding)
    mom4_format : str
        4-momentum format in HDF5 files
    jet_name : str
        Jet name in H5 files ("jet1" or "jet2")
    train_val_split : float, optional
        If provided (e.g., 0.8), will split the same data into train/val
        with this fraction for training. Ignores h5_files_val in this case.
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of workers for data loading
        
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    if jet_name in ["jet1", "jet2"]:
        if train_val_split is not None:
            # Case where the jet is either jet1 or jet2 and data is split into train/val
            # Load all data and split
            print(f"Loading data from H5 files and splitting {train_val_split:.1%}/{1-train_val_split:.1%} train/val...")

            all_features = []
            all_labels = []

            for filename, n_jets in zip(h5_files_train, n_jets_train):
                features, labels = load_lhco_jets_from_h5(
                    h5_filename=filename,
                    feature_dict=feature_dict,
                    n_jets=n_jets,
                    jet_name=jet_name,
                    mom4_format=mom4_format,
                    **kwargs
                )
                # if filename has "supp" in it, change the labels to all 1s (signal)
                if "supp" in filename:
                    labels = np.ones_like(labels)
                all_features.append(features)
                all_labels.append(labels)

            # Concatenate all files
            all_features = ak.concatenate(all_features)
            # Concatenate labels
            all_labels = np.concatenate(all_labels)
            
            # Calculate split index
            n_total = len(all_features)
            n_train = int(n_total * train_val_split)

            print(f"Total jets loaded: {n_total}, Training jets: {n_train}, Validation jets: {n_total - n_train}")
            
            # Shuffle before splitting
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            print(f"Splitting {n_total} jets into {n_train} train / {n_total - n_train} val")
            
            # Split the data
            train_features = all_features[train_indices]
            val_features = all_features[val_indices]
            train_labels = all_labels[train_indices]
            val_labels = all_labels[val_indices]

        else:
            # Case where jet is either jet1 or jet2 and separate train/val files
            raise NotImplementedError("Only train_val_split is supported in this custom loader.")
        
        # Pad and convert to numpy
        train_features_padded, train_mask = ak_pad(
            train_features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded, val_mask = ak_pad(
            val_features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack training features
        train_features_stacked = ak.concatenate(
            [train_features_padded[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features
        val_features_stacked = ak.concatenate(
            [val_features_padded[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        
        # Convert to numpy then to torch tensors
        train_x = torch.from_numpy(ak.to_numpy(train_features_stacked)).float()
        train_mask_t = torch.from_numpy(ak.to_numpy(train_mask)).float()
        # Labels are now just a simple numpy array (0 or 1)
        train_labels_t = torch.from_numpy(train_labels).long()
        
        val_x = torch.from_numpy(ak.to_numpy(val_features_stacked)).float()
        val_mask_t = torch.from_numpy(ak.to_numpy(val_mask)).float()
        # Labels are now just a simple numpy array (0 or 1)
        val_labels_t = torch.from_numpy(val_labels).long()
        
        print(f"Training data shape: {train_x.shape}, Labels: {train_labels_t.shape}")
        print(f"  - Signal: {train_labels_t.sum()}, Background: {(1-train_labels_t).sum()}")
        print(f"Validation data shape: {val_x.shape}, Labels: {val_labels_t.shape}")
        print(f"  - Signal: {val_labels_t.sum()}, Background: {(1-val_labels_t).sum()}")
    
    elif jet_name == "both":
        if train_val_split is not None:
            # Case where both jets are loaded and data is split into train/val
            # Load all data and split
            print(f"Loading data from H5 files and splitting {train_val_split:.1%}/{1-train_val_split:.1%} train/val...")

            all_features_jet1 = []
            all_features_jet2 = []
            all_labels = []

            for filename, n_jets in zip(h5_files_train, n_jets_train):
                features_jet1, features_jet2, labels = load_lhco_jets_from_h5(
                    h5_filename=filename,
                    feature_dict=feature_dict,
                    n_jets=n_jets,
                    jet_name=jet_name,
                    mom4_format=mom4_format,
                    **kwargs
                )
                # if filename has "supp" in it, change the labels to all 1s (signal)
                if "supp" in filename:
                    labels = np.ones_like(labels)
                all_features_jet1.append(features_jet1)
                all_features_jet2.append(features_jet2)
                all_labels.append(labels)

            # Concatenate all files
            all_features_jet1 = ak.concatenate(all_features_jet1)
            all_features_jet2 = ak.concatenate(all_features_jet2)
            # Concatenate labels
            all_labels = np.concatenate(all_labels)
            
            # Calculate split index
            n_total = len(all_features_jet1)
            n_train = int(n_total * train_val_split)

            print(f"Total jets loaded: {n_total}, Training jets: {n_train}, Validation jets: {n_total - n_train}")
            
            # Shuffle before splitting
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            print(f"Splitting {n_total} jets into {n_train} train / {n_total - n_train} val")
            
            # Split the data into train and val sets for both jets
            train_features_jet1 = all_features_jet1[train_indices]
            val_features_jet1 = all_features_jet1[val_indices]
            train_features_jet2 = all_features_jet2[train_indices]
            val_features_jet2 = all_features_jet2[val_indices]
            train_labels = all_labels[train_indices]
            val_labels = all_labels[val_indices]

        else:
            # Case where both jets are loaded and separate train/val files
            raise NotImplementedError("Only train_val_split is supported in this custom loader.")
        
        # Pad and convert to numpy for jet1
        train_features_padded_jet1, train_mask_jet1 = ak_pad(
            train_features_jet1, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded_jet1, val_mask_jet1 = ak_pad(
            val_features_jet1, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )

        # Pad and convert to numpy for jet2
        train_features_padded_jet2, train_mask_jet2 = ak_pad(
            train_features_jet2, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        val_features_padded_jet2, val_mask_jet2 = ak_pad(
            val_features_jet2, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack training features for jet1
        train_features_stacked_jet1 = ak.concatenate(
            [train_features_padded_jet1[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features for jet1
        val_features_stacked_jet1 = ak.concatenate(
            [val_features_padded_jet1[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )

        # Stack training features for jet2
        train_features_stacked_jet2 = ak.concatenate(
            [train_features_padded_jet2[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        # Stack validation features for jet2
        val_features_stacked_jet2 = ak.concatenate(
            [val_features_padded_jet2[feat][..., np.newaxis] for feat in feature_names],
            axis=-1
        )
        
        # Convert to numpy then to torch tensors
        train_x_jet1 = torch.from_numpy(ak.to_numpy(train_features_stacked_jet1)).float()
        train_mask_t_jet1 = torch.from_numpy(ak.to_numpy(train_mask_jet1)).float()
        # Labels are now just a simple numpy array (0 or 1)
        train_labels_t = torch.from_numpy(train_labels).long()
        
        val_x_jet1 = torch.from_numpy(ak.to_numpy(val_features_stacked_jet1)).float()
        val_mask_t_jet1 = torch.from_numpy(ak.to_numpy(val_mask_jet1)).float()
        # Labels are now just a simple numpy array (0 or 1)
        val_labels_t = torch.from_numpy(val_labels).long()

        train_x_jet2 = torch.from_numpy(ak.to_numpy(train_features_stacked_jet2)).float()
        train_mask_t_jet2 = torch.from_numpy(ak.to_numpy(train_mask_jet2)).float()

        val_x_jet2 = torch.from_numpy(ak.to_numpy(val_features_stacked_jet2)).float()
        val_mask_t_jet2 = torch.from_numpy(ak.to_numpy(val_mask_jet2)).float()
        
        
        print(f"Training data shape: {train_x_jet1.shape}, Labels: {train_labels_t.shape}")
        print(f"  - Signal: {train_labels_t.sum()}, Background: {(1-train_labels_t).sum()}")
        print(f"Validation data shape: {val_x_jet1.shape}, Labels: {val_labels_t.shape}")
        print(f"  - Signal: {val_labels_t.sum()}, Background: {(1-val_labels_t).sum()}")


    # Create custom dataset class
    class JetDataset(Dataset):
        def __init__(self, features, masks, labels, features_jet2=None, masks_jet2=None):
            self.features = features
            self.features_jet2 = features_jet2
            self.masks = masks
            self.masks_jet2 = masks_jet2
            self.labels = labels
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            result = {
                "part_features": self.features[idx],
                "part_mask": self.masks[idx],
                "jet_type_labels": self.labels[idx],
                "jet_features": torch.tensor([]),
            }
            # Only include jet2 fields if they exist (for dijet mode)
            if self.features_jet2 is not None:
                result["part_features_jet2"] = self.features_jet2[idx]
                result["part_mask_jet2"] = self.masks_jet2[idx]
            return result
    
    if jet_name in ["jet1", "jet2"]:
        # Create datasets
        train_dataset = JetDataset(features=train_x, masks=train_mask_t, labels=train_labels_t)
        val_dataset = JetDataset(features=val_x, masks=val_mask_t, labels=val_labels_t)

    elif jet_name == "both":
        # Create datasets with both jets
        train_dataset = JetDataset(features=train_x_jet1, masks=train_mask_t_jet1, 
                                   labels=train_labels_t, features_jet2=train_x_jet2, masks_jet2=train_mask_t_jet2)
        val_dataset = JetDataset(features=val_x_jet1, masks=val_mask_t_jet1,
                                 labels=val_labels_t, features_jet2=val_x_jet2, masks_jet2=val_mask_t_jet2)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def create_lhco_h5_test_loader(
        h5_files_test,
        feature_dict,
        batch_size=64,
        n_jets_test=None,
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name="jet1",
        shuffle_test=True,
        num_workers=1,
):
    """Create PyTorch DataLoader from LHCO HDF5 files for testing.
    
    Parameters
    ----------
    h5_files_test : list
        List of HDF5 file paths for testing (e.g., [background_file, signal_file])
    feature_dict : dict
        Feature preprocessing dictionary
    batch_size : int
        Batch size for dataloader
    n_jets_test : int or list of int, optional
        Number of jets to load for testing.
        If int, total number split across files.
        If list, number per file (must match len(h5_files_test)).
        If None, load all jets.
    max_sequence_len : int
        Maximum sequence length (padding)
    mom4_format : str
        4-momentum format in HDF5 files
    jet_name : str
        Jet name in H5 files ("jet1", "jet2", or "both")
    shuffle_test : bool
        Whether to shuffle test data
    num_workers : int
        Number of workers for data loading
        
    Returns
    -------
    test_loader : DataLoader
        Test dataloader
    """    
    # Load all test data from multiple files
    if jet_name in ["jet1", "jet2"]:
        test_features, test_labels = load_multiple_h5_files(
            h5_files_test,
            feature_dict,
            n_jets_per_file=n_jets_test,
            mom4_format=mom4_format,
            jet_name=jet_name,
        )
        
        n_total = len(test_features)
        print(f"Total test jets loaded: {n_total}")
        
        # Shuffle the test data if requested
        if shuffle_test:
            indices = np.random.permutation(n_total)
            test_features = test_features[indices]
            test_labels = test_labels[indices]
        
        # Pad and convert to numpy
        test_features_padded, test_mask = ak_pad(
            test_features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack test features
        test_features_stacked = ak.concatenate(
            [test_features_padded[feat][..., np.newaxis] for feat in feature_names],
            axis=-1,
        )
        
        # Convert to numpy arrays
        test_x = ak.to_numpy(test_features_stacked)
        test_mask_np = ak.to_numpy(test_mask)
        
        # Convert to torch tensors
        test_x = torch.from_numpy(test_x).float()
        test_mask_t = torch.from_numpy(test_mask_np).bool()
        test_labels_t = torch.from_numpy(test_labels).long()
        
        print(f"Test data shape: {test_x.shape}")
        print(f"Test mask shape: {test_mask_t.shape}")
        print(f"Test labels shape: {test_labels_t.shape}")
        print(f"Label distribution: Background (0): {(test_labels == 0).sum()}, Signal (1): {(test_labels == 1).sum()}")
        
        # Create a simple Dataset class
        class JetDataset(torch.utils.data.Dataset):
            def __init__(self, features, masks, labels):
                self.features = features
                self.masks = masks
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    "part_features": self.features[idx],
                    "part_mask": self.masks[idx],
                    "jet_type_labels": self.labels[idx],
                    "jet_features": torch.tensor([]),
                }
        
        # Create test dataset
        test_dataset = JetDataset(test_x, test_mask_t, test_labels_t)
        
    elif jet_name == "both":
        # Load both jets
        test_features_jet1, test_features_jet2, test_labels = load_multiple_h5_files(
            h5_files_test,
            feature_dict,
            n_jets_per_file=n_jets_test,
            mom4_format=mom4_format,
            jet_name=jet_name,
        )
        
        n_total = len(test_features_jet1)
        print(f"Total test jets loaded: {n_total}")
        
        # Shuffle the test data if requested
        if shuffle_test:
            indices = np.random.permutation(n_total)
            test_features_jet1 = test_features_jet1[indices]
            test_features_jet2 = test_features_jet2[indices]
            test_labels = test_labels[indices]
        
        # Pad and convert to numpy for jet1
        test_features_padded_jet1, test_mask_jet1 = ak_pad(
            test_features_jet1, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Pad and convert to numpy for jet2
        test_features_padded_jet2, test_mask_jet2 = ak_pad(
            test_features_jet2, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
        )
        
        # Stack fields into a single array (n_jets, max_sequence_len, n_features)
        feature_names = list(feature_dict.keys())
        
        # Stack test features for jet1
        test_features_stacked_jet1 = ak.concatenate(
            [test_features_padded_jet1[feat][..., np.newaxis] for feat in feature_names],
            axis=-1,
        )
        
        # Stack test features for jet2
        test_features_stacked_jet2 = ak.concatenate(
            [test_features_padded_jet2[feat][..., np.newaxis] for feat in feature_names],
            axis=-1,
        )
        
        # Convert to numpy arrays
        test_x_jet1 = ak.to_numpy(test_features_stacked_jet1)
        test_mask_np_jet1 = ak.to_numpy(test_mask_jet1)
        
        test_x_jet2 = ak.to_numpy(test_features_stacked_jet2)
        test_mask_np_jet2 = ak.to_numpy(test_mask_jet2)
        
        # Convert to torch tensors
        test_x_jet1 = torch.from_numpy(test_x_jet1).float()
        test_mask_t_jet1 = torch.from_numpy(test_mask_np_jet1).bool()
        
        test_x_jet2 = torch.from_numpy(test_x_jet2).float()
        test_mask_t_jet2 = torch.from_numpy(test_mask_np_jet2).bool()
        
        test_labels_t = torch.from_numpy(test_labels).long()
        
        print(f"Test data shape (jet1): {test_x_jet1.shape}")
        print(f"Test data shape (jet2): {test_x_jet2.shape}")
        print(f"Test mask shape (jet1): {test_mask_t_jet1.shape}")
        print(f"Test mask shape (jet2): {test_mask_t_jet2.shape}")
        print(f"Test labels shape: {test_labels_t.shape}")
        print(f"Label distribution: Background (0): {(test_labels == 0).sum()}, Signal (1): {(test_labels == 1).sum()}")
        
        # Create a Dataset class for dijet
        class DijetDataset(torch.utils.data.Dataset):
            def __init__(self, features_jet1, features_jet2, masks_jet1, masks_jet2, labels):
                self.features_jet1 = features_jet1
                self.features_jet2 = features_jet2
                self.masks_jet1 = masks_jet1
                self.masks_jet2 = masks_jet2
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    "part_features":      self.features_jet1[idx],
                    "part_features_jet2": self.features_jet2[idx],
                    "part_mask":          self.masks_jet1[idx],
                    "part_mask_jet2":     self.masks_jet2[idx],
                    "jet_type_labels":    self.labels[idx],
                    "jet_features":       torch.tensor([]),
                }
        
        # Create test dataset
        test_dataset = DijetDataset(test_x_jet1, test_x_jet2, test_mask_t_jet1, test_mask_t_jet2, test_labels_t)
    
    else:
        raise ValueError(f"jet_name must be 'jet1', 'jet2', or 'both', got {jet_name}")
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader


# ---------------------------------------------------------------------------
# CASE dataset loaders
# ---------------------------------------------------------------------------


def create_case_h5_dataloaders(
    h5_files_train,
    h5_files_val,
    feature_dict,
    batch_size=64,
    n_jets_train=None,
    n_jets_val=None,
    max_sequence_len=128,
    jet_name="jet1",
    train_val_split=None,
    shuffle_train=True,
    num_workers=1,
):
    """Create PyTorch DataLoaders from CASE HDF5 files.

    Parameters
    ----------
    h5_files_train : list of str
        HDF5 file paths for training (background files recommended).
    h5_files_val : list of str
        HDF5 file paths for validation.  Ignored when *train_val_split* is set.
    feature_dict : dict
        Feature preprocessing dictionary.
    batch_size : int
        Batch size.
    n_jets_train : int or list of int, optional
        Events to load per training file.
    n_jets_val : int or list of int, optional
        Events to load per validation file.
    max_sequence_len : int
        Padding length for PF-candidate sequences.
    jet_name : str
        ``"jet1"``, ``"jet2"``, or ``"both"``.
    train_val_split : float, optional
        Fraction of *h5_files_train* data used for training.  The remainder
        becomes validation.  *h5_files_val* is ignored in this case.
    shuffle_train : bool
    num_workers : int

    Returns
    -------
    train_loader, val_loader : DataLoader
    """

    class _JetDataset(Dataset):
        def __init__(self, features, masks, labels, features_jet2=None, masks_jet2=None):
            self.features = features
            self.features_jet2 = features_jet2
            self.masks = masks
            self.masks_jet2 = masks_jet2
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            out = {
                "part_features": self.features[idx],
                "part_mask": self.masks[idx],
                "jet_type_labels": self.labels[idx],
                "jet_features": torch.tensor([]),
            }
            if self.features_jet2 is not None:
                out["part_features_jet2"] = self.features_jet2[idx]
                out["part_mask_jet2"] = self.masks_jet2[idx]
            return out

    def _to_tensors(features_ak, labels_np, feature_names, max_seq_len):
        padded, mask = ak_pad(
            features_ak, maxlen=max_seq_len, axis=1, fill_value=0.0, return_mask=True
        )
        stacked = ak.concatenate(
            [padded[f][..., np.newaxis] for f in feature_names], axis=-1
        )
        x = torch.from_numpy(ak.to_numpy(stacked)).float()
        m = torch.from_numpy(ak.to_numpy(mask)).float()
        y = torch.from_numpy(labels_np).long()
        return x, m, y

    feature_names = list(feature_dict.keys())

    if jet_name in ["jet1", "jet2"]:
        if train_val_split is not None:
            print(
                f"Loading CASE data and splitting "
                f"{train_val_split:.1%}/{1-train_val_split:.1%} train/val..."
            )
            all_feats, all_labels = load_multiple_case_h5_files(
                h5_files_train, feature_dict,
                n_jets_per_file=n_jets_train, jet_name=jet_name,
            )
            n_total = len(all_feats)
            n_train = int(n_total * train_val_split)
            idx = np.random.permutation(n_total)
            train_feats = all_feats[idx[:n_train]]
            val_feats   = all_feats[idx[n_train:]]
            train_labels = all_labels[idx[:n_train]]
            val_labels   = all_labels[idx[n_train:]]
        else:
            train_feats, train_labels = load_multiple_case_h5_files(
                h5_files_train, feature_dict,
                n_jets_per_file=n_jets_train, jet_name=jet_name,
            )
            val_feats, val_labels = load_multiple_case_h5_files(
                h5_files_val, feature_dict,
                n_jets_per_file=n_jets_val, jet_name=jet_name,
            )

        train_x, train_m, train_y = _to_tensors(train_feats, train_labels, feature_names, max_sequence_len)
        val_x,   val_m,   val_y   = _to_tensors(val_feats,   val_labels,   feature_names, max_sequence_len)

        print(f"Train: {train_x.shape}  sig={train_y.sum()}  bg={(train_y==0).sum()}")
        print(f"Val:   {val_x.shape}    sig={val_y.sum()}    bg={(val_y==0).sum()}")

        train_dataset = _JetDataset(train_x, train_m, train_y)
        val_dataset   = _JetDataset(val_x,   val_m,   val_y)

    elif jet_name == "both":
        if train_val_split is not None:
            print(
                f"Loading CASE dijet data and splitting "
                f"{train_val_split:.1%}/{1-train_val_split:.1%} train/val..."
            )
            j1_all, j2_all, all_labels = load_multiple_case_h5_files(
                h5_files_train, feature_dict,
                n_jets_per_file=n_jets_train, jet_name="both",
            )
            n_total = len(j1_all)
            n_train = int(n_total * train_val_split)
            idx = np.random.permutation(n_total)
            train_j1 = j1_all[idx[:n_train]]; val_j1 = j1_all[idx[n_train:]]
            train_j2 = j2_all[idx[:n_train]]; val_j2 = j2_all[idx[n_train:]]
            train_labels = all_labels[idx[:n_train]]
            val_labels   = all_labels[idx[n_train:]]
        else:
            train_j1, train_j2, train_labels = load_multiple_case_h5_files(
                h5_files_train, feature_dict,
                n_jets_per_file=n_jets_train, jet_name="both",
            )
            val_j1, val_j2, val_labels = load_multiple_case_h5_files(
                h5_files_val, feature_dict,
                n_jets_per_file=n_jets_val, jet_name="both",
            )

        train_x1, train_m1, train_y = _to_tensors(train_j1, train_labels, feature_names, max_sequence_len)
        train_x2, train_m2, _       = _to_tensors(train_j2, train_labels, feature_names, max_sequence_len)
        val_x1,   val_m1,   val_y   = _to_tensors(val_j1,   val_labels,   feature_names, max_sequence_len)
        val_x2,   val_m2,   _       = _to_tensors(val_j2,   val_labels,   feature_names, max_sequence_len)

        print(f"Train: {train_x1.shape}  sig={train_y.sum()}  bg={(train_y==0).sum()}")
        print(f"Val:   {val_x1.shape}    sig={val_y.sum()}    bg={(val_y==0).sum()}")

        train_dataset = _JetDataset(train_x1, train_m1, train_y, features_jet2=train_x2, masks_jet2=train_m2)
        val_dataset   = _JetDataset(val_x1,   val_m1,   val_y,   features_jet2=val_x2,   masks_jet2=val_m2)

    else:
        raise ValueError(f"jet_name must be 'jet1', 'jet2', or 'both', got {jet_name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def create_case_h5_test_loader(
    h5_files_test,
    feature_dict,
    batch_size=64,
    n_jets_test=None,
    max_sequence_len=128,
    jet_name="jet1",
    shuffle_test=False,
    num_workers=1,
):
    """Create a PyTorch DataLoader from CASE HDF5 files for evaluation.

    Parameters
    ----------
    h5_files_test : list of str
        HDF5 files to load (can mix background and signal files).
    feature_dict : dict
        Feature preprocessing dictionary.
    batch_size : int
        Batch size for dataloader
    n_jets_test : int or list of int, optional
        Events to load per file.
    max_sequence_len : int
        Maximum sequence length (padding)
    jet_name : str
        "jet1", "jet2", or "both".
    shuffle_test : bool
        Whether to shuffle test data
    num_workers : int
        Number of workers for data loading

    Returns
    -------
    test_loader : DataLoader
    """
    feature_names = list(feature_dict.keys())

    def _to_tensors(features_ak, labels_np, max_seq_len):
        padded, mask = ak_pad(
            features_ak, maxlen=max_seq_len, axis=1, fill_value=0.0, return_mask=True
        )
        stacked = ak.concatenate(
            [padded[f][..., np.newaxis] for f in feature_names], axis=-1
        )
        x = torch.from_numpy(ak.to_numpy(stacked)).float()
        m = torch.from_numpy(ak.to_numpy(mask)).bool()
        y = torch.from_numpy(labels_np).long()
        return x, m, y

    if jet_name in ["jet1", "jet2"]:
        feats, labels = load_multiple_case_h5_files(
            h5_files_test, feature_dict,
            n_jets_per_file=n_jets_test, jet_name=jet_name,
        )

        if shuffle_test:
            idx = np.random.permutation(len(feats))
            feats = feats[idx]; labels = labels[idx]

        test_x, test_m, test_y = _to_tensors(feats, labels, max_sequence_len)
        print(f"Test: {test_x.shape}  bg={(test_y==0).sum()}  sig={test_y.sum()}")

        class _JetDataset(Dataset):
            def __init__(self, x, m, y):
                self.x, self.m, self.y = x, m, y
            def __len__(self): return len(self.y)
            def __getitem__(self, idx):
                return {
                    "part_features": self.x[idx],
                    "part_mask": self.m[idx],
                    "jet_type_labels": self.y[idx],
                    "jet_features": torch.tensor([]),
                }

        test_dataset = _JetDataset(test_x, test_m, test_y)

    elif jet_name == "both":
        j1, j2, labels = load_multiple_case_h5_files(
            h5_files_test, feature_dict,
            n_jets_per_file=n_jets_test, jet_name="both",
        )
        if shuffle_test:
            idx = np.random.permutation(len(j1))
            j1 = j1[idx]; j2 = j2[idx]; labels = labels[idx]

        test_x1, test_m1, test_y = _to_tensors(j1, labels, max_sequence_len)
        test_x2, test_m2, _      = _to_tensors(j2, labels, max_sequence_len)
        print(f"Test: {test_x1.shape}  bg={(test_y==0).sum()}  sig={test_y.sum()}")

        class _DijetDataset(Dataset):
            def __init__(self, x1, x2, m1, m2, y):
                self.x1, self.x2, self.m1, self.m2, self.y = x1, x2, m1, m2, y
            def __len__(self): return len(self.y)
            def __getitem__(self, idx):
                return {
                    "part_features":      self.x1[idx],
                    "part_features_jet2": self.x2[idx],
                    "part_mask":          self.m1[idx],
                    "part_mask_jet2":     self.m2[idx],
                    "jet_type_labels":    self.y[idx],
                    "jet_features":       torch.tensor([]),
                }

        test_dataset = _DijetDataset(test_x1, test_x2, test_m1, test_m2, test_y)

    else:
        raise ValueError(f"jet_name must be 'jet1', 'jet2', or 'both', got {jet_name}")

    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )