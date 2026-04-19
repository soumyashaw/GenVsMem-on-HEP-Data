"""Data caching utilities to speed up data loading from H5 files.

This module provides functions to cache preprocessed data as pickle files,
avoiding the slow H5 file loading and preprocessing on subsequent runs.
"""

import os
import pickle
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from gabbro.data.data_utils import create_lhco_h5_dataloaders


def get_cache_key(
    h5_files_train,
    feature_dict,
    n_jets_train,
    max_sequence_len,
    mom4_format,
    jet_name,
    train_val_split,
    batch_size,
):
    """Generate a unique cache key based on data loading parameters.
    
    Parameters
    ----------
    h5_files_train : list
        List of H5 file paths
    feature_dict : dict
        Feature preprocessing dictionary
    n_jets_train : list
        Number of jets per class
    max_sequence_len : int
        Maximum sequence length
    mom4_format : str
        Momentum format
    jet_name : str
        Jet name ("jet1", "jet2", or "both")
    train_val_split : float
        Train/validation split ratio
    batch_size : int
        Batch size
        
    Returns
    -------
    str
        Unique cache key (hash)
    """
    # Create a dictionary of parameters that affect the data
    params = {
        "h5_files": sorted([os.path.basename(f) for f in h5_files_train]),
        "feature_dict": {k: v for k, v in feature_dict.items() if k != "func" and k != "inv_func"},
        "n_jets_train": n_jets_train,
        "max_sequence_len": max_sequence_len,
        "mom4_format": mom4_format,
        "jet_name": jet_name,
        "train_val_split": train_val_split,
    }
    
    # Convert to JSON string and hash it
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    
    return cache_key


def get_cache_path(cache_key, cache_dir="cache"):
    """Get the full path for a cache file.
    
    Parameters
    ----------
    cache_key : str
        Unique cache key
    cache_dir : str
        Cache directory path
        
    Returns
    -------
    Path
        Full path to cache file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"data_cache_{cache_key}.pkl"


def save_cache(cache_path, data_dict):
    """Save preprocessed data to cache file.
    
    Parameters
    ----------
    cache_path : Path
        Path to cache file
    data_dict : dict
        Dictionary containing preprocessed data
    """
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved successfully ({cache_path.stat().st_size / 1024**2:.1f} MB)")


def load_cache(cache_path):
    """Load preprocessed data from cache file.
    
    Parameters
    ----------
    cache_path : Path
        Path to cache file
        
    Returns
    -------
    dict
        Dictionary containing preprocessed data
    """
    print(f"Loading data from cache: {cache_path}...")
    with open(cache_path, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Cache loaded successfully ({cache_path.stat().st_size / 1024**2:.1f} MB)")
    return data_dict


def create_cached_lhco_h5_dataloaders(
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
    cache_dir="cache",
    use_cache=True,
):
    """Create PyTorch DataLoaders with caching support.
    
    This is a wrapper around create_lhco_h5_dataloaders that caches the
    preprocessed data to disk. On first run, it loads from H5 files and
    saves a cache. On subsequent runs with the same parameters, it loads
    from the cache, which is much faster.
    
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
        Number of jets to load for training
    n_jets_val : int or list of int, optional
        Number of jets to load for validation
    max_sequence_len : int
        Maximum sequence length (padding)
    mom4_format : str
        4-momentum format in HDF5 files
    jet_name : str
        Jet name in H5 files ("jet1", "jet2", or "both")
    train_val_split : float, optional
        If provided, will split the same data into train/val
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of workers for data loading
    cache_dir : str
        Directory to store cache files
    use_cache : bool
        Whether to use caching (set to False to force reload from H5)
        
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    # Generate cache key
    cache_key = get_cache_key(
        h5_files_train=h5_files_train,
        feature_dict=feature_dict,
        n_jets_train=n_jets_train,
        max_sequence_len=max_sequence_len,
        mom4_format=mom4_format,
        jet_name=jet_name,
        train_val_split=train_val_split,
        batch_size=batch_size,
    )
    
    cache_path = get_cache_path(cache_key, cache_dir)
    
    # Check if cache exists and use_cache is True
    if use_cache and cache_path.exists():
        # Load from cache
        data_dict = load_cache(cache_path)
        
        # Recreate datasets from cached tensors
        if jet_name in ["jet1", "jet2"]:
            train_dataset = JetDataset(
                features=data_dict["train_features"],
                masks=data_dict["train_masks"],
                labels=data_dict["train_labels"],
            )
            val_dataset = JetDataset(
                features=data_dict["val_features"],
                masks=data_dict["val_masks"],
                labels=data_dict["val_labels"],
            )
        elif jet_name == "both":
            train_dataset = JetDataset(
                features=data_dict["train_features_jet1"],
                masks=data_dict["train_masks_jet1"],
                labels=data_dict["train_labels"],
                features_jet2=data_dict["train_features_jet2"],
                masks_jet2=data_dict["train_masks_jet2"],
            )
            val_dataset = JetDataset(
                features=data_dict["val_features_jet1"],
                masks=data_dict["val_masks_jet1"],
                labels=data_dict["val_labels"],
                features_jet2=data_dict["val_features_jet2"],
                masks_jet2=data_dict["val_masks_jet2"],
            )
        
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
        
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Validation data: {len(val_dataset)} samples")
        
    else:
        # Load from H5 files using original function
        if not use_cache:
            print("Cache disabled, loading from H5 files...")
        else:
            print(f"Cache not found at {cache_path}, loading from H5 files...")
        
        train_loader, val_loader = create_lhco_h5_dataloaders(
            h5_files_train=h5_files_train,
            h5_files_val=h5_files_val,
            feature_dict=feature_dict,
            batch_size=batch_size,
            n_jets_train=n_jets_train,
            n_jets_val=n_jets_val,
            max_sequence_len=max_sequence_len,
            mom4_format=mom4_format,
            jet_name=jet_name,
            train_val_split=train_val_split,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
        )
        
        # Extract tensors from the dataloaders to cache them
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Build cache dictionary
        if jet_name in ["jet1", "jet2"]:
            data_dict = {
                "train_features": train_dataset.features,
                "train_masks": train_dataset.masks,
                "train_labels": train_dataset.labels,
                "val_features": val_dataset.features,
                "val_masks": val_dataset.masks,
                "val_labels": val_dataset.labels,
            }
        elif jet_name == "both":
            data_dict = {
                "train_features_jet1": train_dataset.features,
                "train_masks_jet1": train_dataset.masks,
                "train_features_jet2": train_dataset.features_jet2,
                "train_masks_jet2": train_dataset.masks_jet2,
                "train_labels": train_dataset.labels,
                "val_features_jet1": val_dataset.features,
                "val_masks_jet1": val_dataset.masks,
                "val_features_jet2": val_dataset.features_jet2,
                "val_masks_jet2": val_dataset.masks_jet2,
                "val_labels": val_dataset.labels,
            }
        
        # Save to cache if caching is enabled
        if use_cache:
            save_cache(cache_path, data_dict)
    
    return train_loader, val_loader


class JetDataset(Dataset):
    """Custom Dataset for jet data."""
    
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


def clear_cache(cache_dir="cache"):
    """Clear all cache files in the cache directory.
    
    Parameters
    ----------
    cache_dir : str
        Cache directory path
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("data_cache_*.pkl"))
        for cache_file in cache_files:
            cache_file.unlink()
            print(f"Deleted: {cache_file}")
        print(f"Cleared {len(cache_files)} cache file(s)")
    else:
        print(f"Cache directory {cache_dir} does not exist")


def list_cache_files(cache_dir="cache"):
    """List all cache files in the cache directory with their sizes.
    
    Parameters
    ----------
    cache_dir : str
        Cache directory path
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("data_cache_*.pkl"))
        if cache_files:
            print(f"Found {len(cache_files)} cache file(s) in {cache_dir}:")
            total_size = 0
            for cache_file in cache_files:
                size_mb = cache_file.stat().st_size / 1024**2
                total_size += size_mb
                print(f"  - {cache_file.name}: {size_mb:.1f} MB")
            print(f"Total cache size: {total_size:.1f} MB")
        else:
            print(f"No cache files found in {cache_dir}")
    else:
        print(f"Cache directory {cache_dir} does not exist")
