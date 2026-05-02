"""Shadow model training for memorization score computation (Step 1 & 2).

Runs one trial (shadow model) of the Feldman-Zhang memorization framework.
Each invocation:
  1. Loads the full training dataset S of n events.
  2. Samples a random m-subset I_k using a deterministic per-trial seed.
  3. Trains a fresh BackboneAachenClassificationLightning model (optionally
     loading a pre-trained backbone) on I_k for MAX_EPOCHS epochs.
  4. Runs inference on ALL n events and saves per-event signal probabilities.

Resume support: if the output file for this trial already exists the script
exits immediately so interrupted HTCondor arrays can be restarted safely.

Usage (single trial):
    python -m src.train.train_mem --trial_id 0

Usage (HTCondor array, one job per trial):
    condor_submit scripts/mem_train.sub
"""

import os
import json
import time
import tempfile
import argparse
import numpy as np
import torch
import lightning as L
from collections.abc import Mapping
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

# gabbro imports
from gabbro.models.backbone import BackboneAachenClassificationLightning
from gabbro.models.backbone_base import BackboneTransformer

# Local data loading
from src.data.cache_data import JetDataset
from gabbro.data.loading import load_multiple_h5_files
from gabbro.utils.arrays import ak_pad
import awkward as ak

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers copied/adapted from AnomalyDetection/src/train/train_custom_aachen.py
# ---------------------------------------------------------------------------

def _to_plain_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(v) for v in obj)
    return obj


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _to_attrdict(obj):
    if isinstance(obj, Mapping):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_attrdict(v) for v in obj)
    return obj


def extract_pretrained_backbone_hparams(ckpt_path):
    """Extract backbone hyperparameters from a checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        return {}

    hparams = ckpt.get("hyper_parameters", {})
    if not hparams:
        hparams = ckpt.get("hparams", {})
    if not hparams:
        hparams = ckpt.get("hparams_dict", {})

    if isinstance(hparams, Mapping):
        hparams = _to_plain_dict(hparams)
    else:
        hparams = {}

    if hparams and "backbone_cfg" in hparams and "embedding_dim" not in hparams:
        if isinstance(hparams["backbone_cfg"], Mapping):
            hparams = _to_plain_dict(hparams["backbone_cfg"])

    if hparams and "model_kwargs" in hparams and "embedding_dim" not in hparams:
        model_kwargs = hparams.get("model_kwargs", {})
        if isinstance(model_kwargs, Mapping) and "backbone_cfg" in model_kwargs:
            hparams = _to_plain_dict(model_kwargs["backbone_cfg"])
        elif isinstance(model_kwargs, Mapping) and "embedding_dim" in model_kwargs:
            hparams = _to_plain_dict(model_kwargs)

    return hparams if isinstance(hparams, dict) else {}


def load_pretrained_backbone(model, ckpt_path, strict=False):
    """Load pre-trained backbone weights into a BackboneAachenClassificationLightning model."""
    print(f"Loading pre-trained backbone weights from: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hparams = extract_pretrained_backbone_hparams(ckpt_path)

    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            backbone_state_dict[key[len("backbone."):]] = value
        elif not any(key.startswith(p) for p in ["class_head", "head", "embedding_projection",
                                                   "encoder_layer", "cross_attn", "fusion_proj"]):
            backbone_state_dict[key] = value

    if not backbone_state_dict:
        backbone_state_dict = state_dict
        print("Warning: No 'backbone.' prefix found. Using all keys from checkpoint.")

    if hparams and "embedding_dim" in hparams:
        try:
            print("Instantiating BackboneTransformer with checkpoint hyperparameters...")
            hparams_for_model = _to_attrdict(hparams)
            ckpt_backbone = BackboneTransformer(**hparams_for_model)
            ckpt_backbone.load_state_dict(backbone_state_dict, strict=strict)
            model.backbone = ckpt_backbone
            print("Successfully replaced model.backbone with checkpoint-configured backbone.")
            return
        except Exception as exc:
            print(f"Warning: Could not rebuild backbone from checkpoint hparams ({exc}). "
                  "Falling back to partial in-place loading.")

    # Partial in-place loading with shape-mismatch tolerance
    current_state = model.backbone.state_dict()
    filtered = {}
    skipped_missing, skipped_shape = [], []
    for key, value in backbone_state_dict.items():
        if key not in current_state:
            skipped_missing.append(key)
        elif current_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
        else:
            filtered[key] = value

    if skipped_missing:
        print(f"Warning: Skipping {len(skipped_missing)} backbone keys not present in model.")
    if skipped_shape:
        print(f"Warning: Skipping {len(skipped_shape)} backbone keys with shape mismatch.")

    effective_strict = strict and not skipped_missing and not skipped_shape
    model.backbone.load_state_dict(filtered, strict=effective_strict)
    print("Backbone weights loaded (partial in-place).")


def set_backbone_requires_grad(model, requires_grad=True):
    n = sum(p.numel() for p in model.backbone.parameters())
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad
    print(f"Backbone {'trainable' if requires_grad else 'frozen'}: {n:,} parameters")


# ---------------------------------------------------------------------------
# Model configuration (mirrors train_full.py)
# ---------------------------------------------------------------------------

def create_model_config(pp_dict, args):
    return {
        "particle_features_dict": pp_dict,
        "embedding_dim": args.embedding_dim,
        "max_sequence_len": 128,
        "n_out_nodes": 2,
        "embed_cfg": OmegaConf.create({
            "type": "continuous_project_add",
            "intermediate_dim": None,
        }),
        "transformer_cfg": OmegaConf.create({
            "dim": args.embedding_dim,
            "n_blocks": 8,
            "norm_after_blocks": True,
            "residual_cfg": {"gate_type": "local", "init_value": 1},
            "attn_cfg": {
                "num_heads": 8,
                "dropout_rate": 0.1,
                "norm_before": True,
                "norm_after": False,
            },
            "mlp_cfg": {
                "dropout_rate": 0.0,
                "norm_before": True,
                "expansion_factor": 4,
                "activation": "GELU",
            },
        }),
        "class_head_hidden_dim": 128,
        "class_head_num_heads": 2,
        "class_head_num_CA_blocks": 2,
        "class_head_num_SA_blocks": 0,
        "class_head_dropout_rate": 0.1,
        "jet_features_input_dim": 0,
        "apply_causal_mask": False,
        "zero_padded_start_particle": False,
        "class_weights": None,  # Removed per experiment design
    }


# ---------------------------------------------------------------------------
# Stable full-dataset loading (no random permutation, race-condition safe)
# ---------------------------------------------------------------------------

def _make_stable_cache_key(signal_path, background_path, n_jets_train,
                            feature_dict, max_sequence_len, mom4_format):
    """Deterministic hash for the memorization full-dataset cache."""
    import hashlib, json
    params = {
        "signal":     os.path.basename(signal_path),
        "background": os.path.basename(background_path),
        "n_jets":     n_jets_train,
        "features":   list(feature_dict.keys()),
        "seq_len":    max_sequence_len,
        "mom4":       mom4_format,
        "version":    "mem_v1",  # bump when loading logic changes
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def load_full_dataset_stable(
    signal_path, background_path, n_jets_train,
    feature_dict, max_sequence_len=128, mom4_format="epxpypz",
    cache_dir="cache",
):
    """Load the complete training dataset in a stable, permutation-free order.

    Signal events are placed first, then background events.  Results are
    cached to disk so that all parallel trials share the same event ordering.
    Atomic write (temp file + os.replace) prevents cache corruption under
    concurrent writes — safe because the data is fully deterministic.

    Parameters
    ----------
    signal_path, background_path : str
        Paths to signal and background H5 files.
    n_jets_train : list[int]
        [n_signal, n_background] events to load.
    feature_dict : dict
        Preprocessing dict (same format as train_full.py).
    max_sequence_len : int
        Padding length.
    mom4_format : str
        4-momentum format in H5 file.
    cache_dir : str
        Directory for cache files.

    Returns
    -------
    JetDataset
        Dataset with all events in stable order (signal first, then background).
    """
    cache_key = _make_stable_cache_key(
        signal_path, background_path, n_jets_train,
        feature_dict, max_sequence_len, mom4_format,
    )
    cache_path = Path(cache_dir) / f"mem_full_{cache_key}.pkl"

    if cache_path.exists():
        print(f"Loading full dataset from stable cache: {cache_path}")
        import pickle
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return JetDataset(
            features=data["features_jet1"],
            masks=data["masks_jet1"],
            labels=data["labels"],
            features_jet2=data["features_jet2"],
            masks_jet2=data["masks_jet2"],
        )

    print("Building stable full-dataset cache (first run only)...")
    features_jet1_ak, features_jet2_ak, labels = load_multiple_h5_files(
        h5_filenames=[signal_path, background_path],
        feature_dict=feature_dict,
        n_jets_per_file=n_jets_train,
        mom4_format=mom4_format,
        jet_name="both",
    )

    feature_names = list(feature_dict.keys())

    def _pad_and_stack(ak_feats):
        padded, mask = ak_pad(ak_feats, maxlen=max_sequence_len,
                              axis=1, fill_value=0.0, return_mask=True)
        stacked = ak.concatenate(
            [padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        x = torch.from_numpy(ak.to_numpy(stacked)).float()
        m = torch.from_numpy(ak.to_numpy(mask)).float()
        return x, m

    feat1, mask1 = _pad_and_stack(features_jet1_ak)
    feat2, mask2 = _pad_and_stack(features_jet2_ak)
    labels_t = torch.from_numpy(labels).long()

    data = {
        "features_jet1": feat1,
        "masks_jet1":    mask1,
        "features_jet2": feat2,
        "masks_jet2":    mask2,
        "labels":        labels_t,
    }

    # Atomic write: write to temp file, then rename (last writer wins, but all
    # write identical content so this is safe).
    import pickle
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.pkl")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        print(f"Stable cache saved: {cache_path} "
              f"({cache_path.stat().st_size / 1024**2:.1f} MB)")
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    return JetDataset(
        features=feat1, masks=mask1, labels=labels_t,
        features_jet2=feat2, masks_jet2=mask2,
    )


# ---------------------------------------------------------------------------
# Full-dataset inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(model, full_dataset, batch_size, device):
    """Run model inference on all n events and return per-event signal probabilities."""
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )
    model.eval()
    all_probs = []
    for batch in loader:
        X1 = batch["part_features"].to(device)
        X2 = batch["part_features_jet2"].to(device)
        mask1 = batch["part_mask"].to(device)
        mask2 = batch["part_mask_jet2"].to(device)
        logits = model(X1, mask1, X2, mask2)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Atomic save: write to a temp file then rename to avoid partial writes
# ---------------------------------------------------------------------------

def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Single-trial runner (called in a loop from main)
# ---------------------------------------------------------------------------

def run_trial(trial_id, args, full_dataset, n, m, model_kwargs_base,
              gpu_id, device, output_dir):
    """Train one shadow model and save predictions. Returns True if ran, False if skipped."""
    trial_file = output_dir / f"trial_{trial_id:04d}.npz"
    if trial_file.exists():
        print(f"  [trial {trial_id}] already done — skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"  Trial {trial_id}  |  device: {device}")
    print(f"{'='*60}")

    trial_seed = args.base_seed + trial_id

    # Sample subset indices deterministically for this trial
    rng = np.random.default_rng(trial_seed)
    subset_indices = rng.choice(n, size=m, replace=False)

    # Build subset DataLoader
    L.seed_everything(trial_seed)
    subset_loader = DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=(gpu_id is not None),
        drop_last=False,
    )

    # Deep-copy model_kwargs so each trial gets its own mutable copy
    import copy
    model_kwargs = copy.deepcopy(model_kwargs_base)

    # Build fresh model
    model = BackboneAachenClassificationLightning(
        optimizer=partial(torch.optim.AdamW, lr=args.learning_rate, weight_decay=1e-2),
        scheduler=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0),
        merge_strategy="concat",
        model_kwargs=model_kwargs,
        use_continuous_input=True,
        scheduler_lightning_kwargs={"interval": "step", "frequency": 1},
    )

    if args.load_pretrained and args.pretrained_ckpt:
        load_pretrained_backbone(model, args.pretrained_ckpt, strict=False)

    if args.freeze_backbone:
        set_backbone_requires_grad(model, requires_grad=False)

    # Train
    trial_log_dir = output_dir / "lightning_logs" / f"trial_{trial_id:04d}"
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if gpu_id is not None else "cpu",
        devices=[gpu_id] if gpu_id is not None else 1,
        callbacks=[],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=50,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        precision="32",
        num_nodes=1,
        default_root_dir=str(trial_log_dir),
    )

    t0 = time.time()
    trainer.fit(model=model, train_dataloaders=subset_loader)
    elapsed = (time.time() - t0) / 60
    print(f"  [trial {trial_id}] training done in {elapsed:.1f} min")

    # Inference on all n events
    model = model.to(device)
    preds = predict_all(model, full_dataset, batch_size=args.batch_size * 2, device=device)

    # Save
    atomic_save_npz(
        trial_file,
        subset_indices=subset_indices.astype(np.int32),
        preds=preds,
    )
    print(f"  [trial {trial_id}] saved → {trial_file}")

    # Explicitly free model memory before next trial
    del model, trainer
    if gpu_id is not None:
        torch.cuda.empty_cache()

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Memorization shadow models — chunk of sequential trials on one GPU"
    )

    # Chunk identity (primary usage)
    parser.add_argument("--chunk_id", type=int, default=None,
                        help="Chunk index (0-based). Trials [chunk_id*chunk_size, "
                             "(chunk_id+1)*chunk_size) are run sequentially.")
    parser.add_argument("--chunk_size", type=int, default=200,
                        help="Number of trials per chunk job (default: 200)")

    # Single-trial fallback (for quick testing)
    parser.add_argument("--trial_id", type=int, default=None,
                        help="Run a single trial instead of a full chunk. "
                             "Overrides --chunk_id / --chunk_size.")

    # Global trial count
    parser.add_argument("--n_trials", type=int, default=4000,
                        help="Total number of trials t")
    parser.add_argument("--subset_fraction", type=float, default=0.7,
                        help="Fraction m/n for each trial subset")
    parser.add_argument("--base_seed", type=int, default=int(os.getenv("SEED", 42)),
                        help="Base random seed; trial k uses seed = base_seed + k")

    # Dataset
    parser.add_argument("--dataset_path", default=str(os.getenv("DATASET_PATH")),
                        help="Path to LHCO dataset directory")
    parser.add_argument("--n_jets_train",
                        type=lambda s: list(map(int, s.strip("[]").split(","))),
                        default=list(map(int,
                            os.getenv("N_JETS_TRAIN", "[25000,25000]").strip("[]").split(","))),
                        help="Number of jets per class [signal, background]")

    # Output
    parser.add_argument("--output_dir",
                        default="/.automount/net_rw/net__data_ttk/soshaw/GenVsMem/models/memorization",
                        help="Directory for per-trial prediction files")

    # Training hyper-parameters
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 64)))
    parser.add_argument("--max_epochs", type=int, default=int(os.getenv("MAX_EPOCHS", 10)))
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE", 1e-4)))
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM", 128)))

    # Pretrained backbone
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to pre-trained backbone checkpoint")
    parser.add_argument("--load_pretrained", action="store_true",
                        help="Load pre-trained backbone weights before fine-tuning")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone — only train the classification head")

    # GPU
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU device ID. Defaults to 0 if GPUs are available.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve trial range
    # ------------------------------------------------------------------
    if args.trial_id is not None:
        # Single-trial mode (testing / debugging)
        trial_start = args.trial_id
        trial_end   = args.trial_id + 1
    elif args.chunk_id is not None:
        trial_start = args.chunk_id * args.chunk_size
        trial_end   = min(trial_start + args.chunk_size, args.n_trials)
    else:
        parser.error("Provide either --chunk_id or --trial_id.")

    # ------------------------------------------------------------------
    # GPU assignment (all trials in this chunk share one GPU)
    # ------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    if args.gpu_id is not None:
        gpu_id = args.gpu_id
    elif n_gpus > 0:
        gpu_id = 0
    else:
        gpu_id = None

    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and n_gpus > 0 else "cpu")

    print(f"Chunk {args.chunk_id}  |  trials {trial_start}–{trial_end - 1}  "
          f"|  device: {device}  |  n_gpus: {n_gpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load (or build) full dataset ONCE for the entire chunk
    # ------------------------------------------------------------------
    input_features_dict = {
        "part_pt":     {"multiply_by": 1, "subtract_by": 1.8,
                        "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3},
    }

    signal_path     = os.path.join(args.dataset_path, "sn_25k_SR_train.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")

    full_dataset = load_full_dataset_stable(
        signal_path=signal_path,
        background_path=background_path,
        n_jets_train=args.n_jets_train,
        feature_dict=input_features_dict,
        max_sequence_len=128,
        mom4_format="epxpypz",
        cache_dir="cache",
    )

    n = len(full_dataset)
    m = int(args.subset_fraction * n)
    print(f"Full dataset n={n}, subset size m={m} ({args.subset_fraction:.0%})")

    # ------------------------------------------------------------------
    # 2. Save dataset labels once (atomic write, safe under concurrency)
    # ------------------------------------------------------------------
    info_file = output_dir / "dataset_info.npz"
    if not info_file.exists():
        lbl = full_dataset.labels
        labels_all = lbl.numpy() if torch.is_tensor(lbl) else np.array(lbl)
        try:
            atomic_save_npz(info_file, labels=labels_all, n_events=np.array(n))
            print(f"Saved dataset info → {info_file}")
        except Exception:
            pass  # Another concurrent job already wrote it — fine

    # ------------------------------------------------------------------
    # 3. Pre-compute model_kwargs (with pretrained hparam override if needed)
    # ------------------------------------------------------------------
    model_kwargs_base = create_model_config(input_features_dict, args)

    if args.load_pretrained and args.pretrained_ckpt:
        pretrained_hparams = extract_pretrained_backbone_hparams(args.pretrained_ckpt)
        if pretrained_hparams and "embedding_dim" in pretrained_hparams:
            backbone_keys = [
                "particle_features_dict", "embedding_dim", "max_sequence_len",
                "n_out_nodes", "embed_cfg", "transformer_cfg",
                "jet_features_input_dim", "apply_causal_mask", "zero_padded_start_particle",
            ]
            for k in backbone_keys:
                if k in pretrained_hparams:
                    model_kwargs_base[k] = pretrained_hparams[k]
            print("Backbone model_kwargs overridden from pretrained checkpoint hparams.")

        for cfg_key in ["embed_cfg", "transformer_cfg"]:
            if cfg_key in model_kwargs_base and isinstance(model_kwargs_base[cfg_key], Mapping):
                model_kwargs_base[cfg_key] = _to_attrdict(model_kwargs_base[cfg_key])

    # ------------------------------------------------------------------
    # 4. Loop over trials in this chunk
    # ------------------------------------------------------------------
    n_ran = 0
    n_skipped = 0
    chunk_t0 = time.time()

    for trial_id in range(trial_start, trial_end):
        ran = run_trial(
            trial_id=trial_id,
            args=args,
            full_dataset=full_dataset,
            n=n,
            m=m,
            model_kwargs_base=model_kwargs_base,
            gpu_id=gpu_id,
            device=device,
            output_dir=output_dir,
        )
        if ran:
            n_ran += 1
        else:
            n_skipped += 1

    chunk_elapsed = (time.time() - chunk_t0) / 60
    print(f"\nChunk done: {n_ran} trained, {n_skipped} skipped, "
          f"total wall time {chunk_elapsed:.1f} min")


if __name__ == "__main__":
    main()
