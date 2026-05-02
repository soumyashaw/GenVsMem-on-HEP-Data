"""Step 4: Compute memorization scores and analyse signal enrichment.

Loads per-trial prediction files produced by src/train/train_mem.py, computes
the Feldman-Zhang memorization score for every training event, and produces:
  - A plot of classification threshold (x-axis) vs fraction of signal events
    among the memorised set (y-axis).
  - A pickle file of highly-memorised event data (features, masks, labels,
    original indices) for downstream analysis.

Memorization score definition:
    mem(i) = Pr[h_k(x_i) = y_i | i ∈ I_k] - Pr[h_k(x_i) = y_i | i ∉ I_k]

where "correct prediction" means sigmoid(logit) > 0.5 matches the true label.

Usage:
    python -m src.eval.compute_memorization \\
        --output_dir /.automount/net_rw/net__data_ttk/soshaw/GenVsMem/models/memorization \\
        --n_trials 4000 \\
        --plot_output memorization_signal_fraction.png \\
        --pkl_output memorized_events.pkl
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_memorization_scores(output_dir: Path, n_trials: int, n_events: int):
    """Accumulate per-trial data and return memorization scores.

    Parameters
    ----------
    output_dir : Path
        Directory containing trial_XXXX.npz files and dataset_info.npz.
    n_trials : int
        Expected total number of trials t.
    n_events : int
        Total number of training events n.

    Returns
    -------
    mem_scores : np.ndarray, shape (n_events,), float32
        Memorization score for each event.
    labels : np.ndarray, shape (n_events,), int
        Ground-truth labels (1 = signal, 0 = background).
    n_loaded : int
        Number of trial files actually loaded (may be < n_trials if incomplete).
    """
    # Load ground-truth labels
    info_file = output_dir / "dataset_info.npz"
    if not info_file.exists():
        raise FileNotFoundError(
            f"dataset_info.npz not found in {output_dir}. "
            "Run at least one trial first to generate it."
        )
    labels = np.load(info_file)["labels"].astype(np.int32)
    assert len(labels) == n_events, (
        f"Labels length {len(labels)} != n_events {n_events}"
    )

    # Accumulators
    in_correct  = np.zeros(n_events, dtype=np.float64)
    out_correct = np.zeros(n_events, dtype=np.float64)
    in_count    = np.zeros(n_events, dtype=np.int32)

    n_loaded = 0
    for k in range(n_trials):
        trial_file = output_dir / f"trial_{k:04d}.npz"
        if not trial_file.exists():
            continue
        data = np.load(trial_file)
        preds = data["preds"].astype(np.float32)     # shape (n_events,)
        subset_idx = data["subset_indices"].astype(np.int32)

        # Binary correct/incorrect for all events
        predicted_label = (preds > 0.5).astype(np.int32)
        correct = (predicted_label == labels).astype(np.float32)

        # In-mask: boolean array indicating which events were in this trial's subset
        in_mask = np.zeros(n_events, dtype=bool)
        in_mask[subset_idx] = True

        in_correct  += correct * in_mask
        out_correct += correct * (~in_mask)
        in_count    += in_mask.astype(np.int32)

        n_loaded += 1
        if n_loaded % 100 == 0:
            print(f"  Processed {n_loaded}/{n_trials} trials...", flush=True)

    print(f"Loaded {n_loaded} trial files.")

    out_count = n_loaded - in_count  # number of trials where event was OUT

    # Avoid division by zero for events that were always in or always out
    with np.errstate(invalid="ignore", divide="ignore"):
        prob_in  = np.where(in_count > 0,  in_correct  / in_count,  0.0)
        prob_out = np.where(out_count > 0, out_correct / out_count, 0.0)

    mem_scores = (prob_in - prob_out).astype(np.float32)
    return mem_scores, labels, n_loaded


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_threshold_vs_signal_fraction(mem_scores, labels, plot_output: Path):
    """Plot memorization threshold (x-axis) vs fraction of signal events in memorised set."""
    thresholds = np.linspace(0.0, 1.0, 201)
    signal_fractions = []
    n_memorised = []

    for tau in thresholds:
        mask = mem_scores >= tau
        total = mask.sum()
        if total == 0:
            signal_fractions.append(np.nan)
        else:
            signal_fractions.append(labels[mask].mean())
        n_memorised.append(total)

    signal_fractions = np.array(signal_fractions)
    n_memorised = np.array(n_memorised)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_frac  = "#1f77b4"
    color_count = "#ff7f0e"

    ax1.plot(thresholds, signal_fractions * 100, color=color_frac, linewidth=2)
    ax1.set_xlabel("Memorization threshold $\\tau$", fontsize=13)
    ax1.set_ylabel("Signal fraction among memorised events (%)", fontsize=12, color=color_frac)
    ax1.tick_params(axis="y", labelcolor=color_frac)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, n_memorised, color=color_count, linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Number of memorised events", fontsize=12, color=color_count)
    ax2.tick_params(axis="y", labelcolor=color_count)

    ax1.set_title(
        "Memorization threshold vs signal fraction in memorised set\n"
        "(dashed: number of events above threshold)",
        fontsize=12,
    )

    fig.tight_layout()
    plot_output = Path(plot_output)
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_output}")


# ---------------------------------------------------------------------------
# Save highly-memorised events
# ---------------------------------------------------------------------------

def save_memorised_events(
    mem_scores,
    labels,
    output_dir: Path,
    pkl_output: Path,
    threshold: float = 0.25,
):
    """Save events with memorization score >= threshold to a pickle file.

    The pickle file contains a dict with keys:
      - indices        : original indices into the full dataset
      - mem_scores     : memorization scores for those events
      - labels         : ground-truth labels (1=signal, 0=background)
      - features_jet1  : particle features for jet 1, shape (N, seq_len, n_feat)
      - masks_jet1     : particle mask for jet 1,     shape (N, seq_len)
      - features_jet2  : particle features for jet 2, shape (N, seq_len, n_feat)
      - masks_jet2     : particle mask for jet 2,     shape (N, seq_len)
    """
    info_file = output_dir / "dataset_info.npz"
    highly_mem_mask = mem_scores >= threshold
    highly_mem_indices = np.where(highly_mem_mask)[0]

    n_total   = len(mem_scores)
    n_hm      = len(highly_mem_indices)
    n_signal  = labels[highly_mem_mask].sum()
    print(f"\nThreshold τ = {threshold}")
    print(f"  Highly memorised events: {n_hm}/{n_total} ({100*n_hm/n_total:.2f}%)")
    print(f"  Signal among them:       {n_signal}/{n_hm} ({100*n_signal/n_hm:.2f}%)")

    # Load raw tensors from the cache (dataset_info.npz holds only labels;
    # full tensors must be reloaded via the existing cached DataLoader).
    # We import here to avoid circular dependency at module level.
    import os
    import torch
    from dotenv import load_dotenv
    load_dotenv()

    from src.data.cache_data import create_cached_lhco_h5_dataloaders

    dataset_path = os.getenv("DATASET_PATH")
    input_features_dict = {
        "part_pt":     {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3},
    }
    n_jets_train_str = os.getenv("N_JETS_TRAIN", "[25000,25000]")
    n_jets_train = list(map(int, n_jets_train_str.strip("[]").split(",")))

    signal_path     = os.path.join(dataset_path, "sn_25k_SR_train.h5")
    background_path = os.path.join(dataset_path, "bg_200k_SR_train.h5")

    loader, _ = create_cached_lhco_h5_dataloaders(
        h5_files_train=[signal_path, background_path],
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=256,
        n_jets_train=n_jets_train,
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name="both",
        train_val_split=1.0,
        shuffle_train=False,
        num_workers=1,
        cache_dir="cache",
        use_cache=True,
    )
    ds = loader.dataset

    def _to_numpy(t):
        if torch.is_tensor(t):
            return t.numpy()
        return np.array(t)

    feat1  = _to_numpy(ds.features)[highly_mem_indices]
    mask1  = _to_numpy(ds.masks)[highly_mem_indices]
    feat2  = _to_numpy(ds.features_jet2)[highly_mem_indices]
    mask2  = _to_numpy(ds.masks_jet2)[highly_mem_indices]

    save_dict = {
        "indices":       highly_mem_indices,
        "mem_scores":    mem_scores[highly_mem_mask],
        "labels":        labels[highly_mem_mask],
        "features_jet1": feat1,
        "masks_jet1":    mask1,
        "features_jet2": feat2,
        "masks_jet2":    mask2,
        "threshold":     threshold,
    }

    pkl_output = Path(pkl_output)
    pkl_output.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_output, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {n_hm} highly-memorised events to {pkl_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_out = "/.automount/net_rw/net__data_ttk/soshaw/GenVsMem/models/memorization"

    parser = argparse.ArgumentParser(description="Compute memorization scores from shadow models")
    parser.add_argument("--output_dir",  default=default_out,
                        help="Directory with trial_XXXX.npz files")
    parser.add_argument("--n_trials",   type=int,   default=4000,
                        help="Total number of trials t")
    parser.add_argument("--n_events",   type=int,   default=None,
                        help="Total training events n. Inferred from dataset_info.npz if omitted.")
    parser.add_argument("--plot_output", default="memorization_signal_fraction.png",
                        help="Path for the threshold-vs-signal-fraction plot")
    parser.add_argument("--pkl_output",  default="memorized_events.pkl",
                        help="Path for the pickle file of highly-memorised events")
    parser.add_argument("--pkl_threshold", type=float, default=0.25,
                        help="Memorization score threshold for saving to pkl (default: 0.25)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Infer n_events from dataset_info.npz if not given
    if args.n_events is None:
        info_file = output_dir / "dataset_info.npz"
        if info_file.exists():
            n_events = int(np.load(info_file)["labels"].shape[0])
        else:
            raise ValueError(
                "--n_events not specified and dataset_info.npz not found. "
                "Run at least one training trial first."
            )
    else:
        n_events = args.n_events

    print(f"Computing memorization scores for {n_events} events over up to {args.n_trials} trials ...")
    mem_scores, labels, n_loaded = compute_memorization_scores(output_dir, args.n_trials, n_events)

    if n_loaded == 0:
        print("No trial files found. Nothing to compute.")
        return

    print(f"\nMemorization score summary (n={n_events}, trials={n_loaded}):")
    print(f"  min={mem_scores.min():.4f}  max={mem_scores.max():.4f}  "
          f"mean={mem_scores.mean():.4f}  std={mem_scores.std():.4f}")

    signal_mask = labels == 1
    bg_mask     = labels == 0
    print(f"  Signal events:     {signal_mask.sum()}")
    print(f"  Background events: {bg_mask.sum()}")
    print(f"  Mean mem score — signal: {mem_scores[signal_mask].mean():.4f} | "
          f"background: {mem_scores[bg_mask].mean():.4f}")

    # Plot
    plot_threshold_vs_signal_fraction(mem_scores, labels, Path(args.plot_output))

    # Save memorised events to pkl
    save_memorised_events(
        mem_scores,
        labels,
        output_dir=output_dir,
        pkl_output=Path(args.pkl_output),
        threshold=args.pkl_threshold,
    )

    # Print a summary table at a few representative thresholds
    print("\nSignal fraction at selected thresholds:")
    print(f"  {'τ':>6}  {'N memorised':>13}  {'N signal':>10}  {'Signal %':>10}")
    for tau in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask  = mem_scores >= tau
        n_mem = mask.sum()
        n_sig = labels[mask].sum()
        frac  = 100.0 * n_sig / n_mem if n_mem > 0 else float("nan")
        print(f"  {tau:>6.2f}  {n_mem:>13,}  {n_sig:>10,}  {frac:>9.2f}%")


if __name__ == "__main__":
    main()
