""" Python file to train the OmniJet-alpha model with continuous tokens - Aachen supervised variant.

PROJECT STRUCTURE:
AnomalyDetection/
├── src/
│   ├── train/              # Training scripts (e.g., train_aachen.py)
│   ├── eval/               # Evaluation scripts (evaluate.py, evaluate_true_roc.py, etc.)
│   ├── data/               # Data processing (split_h5_dataset.py)
│   ├── viz/                # Visualization (unsupervised_learning.py, visualize_clustering.py)
│   └── poc_expts/          # Proof-of-concept experiments
├── scripts/                # Job launchers (.sh and .sub files)
├── gabbro/                 # Core library (models, data utilities, metrics)
└── [output directories: plots/, results/, logs/, checkpoints/, dijet_expts/, aachen_head_expts/]

COMPONENTS USED:
├── Data Loading & Preprocessing
│   ├── load_multiple_h5_files
│   ├── create_lhco_h5_dataloaders
│   └── ak_pad
│
├── Model Architecture
│   ├── BackboneAachenClassificationLightning (two-jet model)
│   │   ├── BackboneTransformer blocks
│   │   └── AachenClassificationHead
│   │
│   └── Loss: CrossEntropyLoss with class weighting
│
└── Training
    ├── PyTorch Lightning Trainer
    ├── AdamW + ConstantLR Scheduler
    ├── AUC & ARGOS metric callbacks
    └── Weights & Biases logging

USAGE: python -m src.train.train_aachen [args] or ./scripts/train_aachen.sh
Training an Aachen-style two-jet anomaly detection model. (Trained on 200k bkg + 25k signal jets per jet) (Tested on 200k bkg + 50k signal jets)
"""
# imports
import os
import json
import torch
import argparse
import numpy as np
import awkward as ak
import lightning as L
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, roc_curve
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import wandb

# gabbro imports
from gabbro.utils.arrays import ak_pad
from gabbro.data.data_utils import create_lhco_h5_dataloaders
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

load_dotenv()  # Load environment variables from .env file (for W&B API key, etc.)

class ExperimentLogger:
    """Handles logging of experiment configuration and results."""
    
    def __init__(self, log_dir="logs", naming_identifier=""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = str(naming_identifier).strip()
        if identifier:
            self.run_name = f"run_{identifier}_{self.timestamp}"
        else:
            self.run_name = f"run_{self.timestamp}"

        # Create run-specific directory
        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config dictionary
        self.config = {}
        self.results = {}
        
    def log_config(self, config_dict):
        """Log experiment configuration."""
        self.config.update(config_dict)
        
        # Save config to JSON
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
        
    def log_results(self, results_dict):
        """Log experiment results."""
        self.results.update(results_dict)
        
        # Save results to JSON
        results_path = self.run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def log_final_results(self, trainer, checkpoint_callback):
        """Log final training results and metrics."""
        final_results = {
            "best_model_path": checkpoint_callback.best_model_path,
            "best_model_score": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "training_completed": True,
            "timestamp_end": datetime.now().isoformat(),
        }
        
        # Add callback metrics if available
        if hasattr(trainer, 'callback_metrics'):
            metrics = {k: float(v) if torch.is_tensor(v) else v 
                      for k, v in trainer.callback_metrics.items()}
            final_results["final_metrics"] = metrics
        
        self.log_results(final_results)
        
        # Create summary log file
        summary_path = self.run_dir / "summary.log"
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.run_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write("CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("RESULTS:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.results.items():
                f.write(f"{key}: {value}\n")
    
    def get_checkpoint_dir(self):
        """Get checkpoint directory for this run."""
        return str(self.run_dir / "checkpoints")


def create_model_config(pp_dict, args):
    """Create model configuration for BackboneTransformer.
    
    Parameters
    ----------
    pp_dict : dict
        Preprocessing dictionary
        
    Returns
    -------
    dict
        Model configuration
    """
    model_kwargs = {
        # Feature specification
        "particle_features_dict": pp_dict,
        
        # Architecture
        "embedding_dim": args.embedding_dim,
        "max_sequence_len": 128,
        "n_out_nodes": 2,  # Binary classification (signal vs background)
        
        "embed_cfg": OmegaConf.create({
            "type": "continuous_project_add",
            "intermediate_dim": None,
        }),
        
        # Transformer configuration
        "transformer_cfg": OmegaConf.create({
            "dim": args.embedding_dim,  # Must match embedding_dim
            "n_blocks": 8,
            "norm_after_blocks": True,
            "residual_cfg": {
                "gate_type": "local",
                "init_value": 1,
            },
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
        
        # # Classification head settings (for class_attention type omnijet alpha)
        # "class_head_hidden_dim": 128,
        # "class_head_num_heads": 8,
        # "class_head_num_CA_blocks": 2,
        # "class_head_num_SA_blocks": 0,
        # "class_head_dropout_rate": 0.1,

        # Anomaly detection head settings (for Aachen method)
        "class_head_hidden_dim": 128,
        "class_head_num_heads": 2,
        "class_head_num_CA_blocks": 2,
        "class_head_num_SA_blocks": 0,
        "class_head_dropout_rate": 0.1,
        
        # Jet-level features
        "jet_features_input_dim": 0,
        
        # Other settings
        "apply_causal_mask": False,
        "zero_padded_start_particle": False,
    }
    
    return model_kwargs


class AUCCallback(Callback):
    """Compute AUC on the validation set at the end of each validation epoch
    and log it to the LightningModule so it becomes part of callback_metrics.
    Note: this will run the model over the validation loader again, so it
    duplicates work performed by the Lightning validation loop (but it ensures
    we have a reliable ROC AUC metric in trainer.callback_metrics and therefore
    in the experiment results.json).
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        # Try to get first validation dataloader
        try:
            val_loaders = trainer.val_dataloaders
        except Exception:
            val_loaders = None
        if not val_loaders:
            return

        # Handle both single DataLoader and list of DataLoaders
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders
            
        device = pl_module.device if hasattr(pl_module, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        all_preds = []
        all_labels = []

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                # expect dict-style batches as used in data_utils
                labels = batch["jet_type_labels"].to(device)
                
                # Check if model is dijet or single-jet
                if isinstance(pl_module, (BackboneDijetClassificationLightning, BackboneAachenClassificationLightning)):
                    # Dijet model: needs both jets
                    X1 = batch["part_features"].to(device)
                    X2 = batch["part_features_jet2"].to(device)
                    mask1 = batch["part_mask"].to(device)
                    mask2 = batch["part_mask_jet2"].to(device)
                    logits = pl_module(X1, mask1, X2, mask2)
                else:
                    # Single-jet model: needs only one jet
                    X = batch["part_features"].to(device)
                    mask = batch["part_mask"].to(device)
                    logits = pl_module(X, mask)
                
                # Handle different logit shapes
                if logits.dim() == 1:
                    # Binary classification with single logit (BCEWithLogitsLoss)
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    # Multi-class with softmax
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # If only one class present, roc_auc_score will fail - handle gracefully
        try:
            auc_val = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc_val = float('nan')

        # Log the metric so Lightning records it in callback_metrics
        pl_module.log("val_auc", auc_val, prog_bar=True, logger=True)

class ARGOSCallback(Callback):
    """Compute ARGOS metric on the validation set at the end of each validation epoch.
    ARGOS is defined as: max(tpr/sqrt(fpr) - sqrt(tpr)) for fpr > 0.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation dataloader
        try:
            val_loaders = trainer.val_dataloaders
        except Exception:
            val_loaders = None
        if not val_loaders:
            return

        # Handle both single DataLoader and list of DataLoaders
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders
            
        device = pl_module.device if hasattr(pl_module, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        all_preds = []
        all_labels = []

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["jet_type_labels"].to(device)
                
                # Check if model is dijet or single-jet
                if isinstance(pl_module, BackboneDijetClassificationLightning):
                    X1 = batch["part_features"].to(device)
                    X2 = batch["part_features_jet2"].to(device)
                    mask1 = batch["part_mask"].to(device)
                    mask2 = batch["part_mask_jet2"].to(device)
                    logits = pl_module(X1, mask1, X2, mask2)
                else:
                    X = batch["part_features"].to(device)
                    mask = batch["part_mask"].to(device)
                    logits = pl_module(X, mask)
                
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # Compute ARGOS metric
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            inds = np.nonzero(fpr)
            tpr = tpr[inds]
            fpr = fpr[inds]
            argos = float(np.max(tpr/np.sqrt(fpr) - np.sqrt(tpr)))
        except Exception:
            argos = float('nan')

        # Log the metric
        pl_module.log("val_argos", argos, prog_bar=True, logger=True)

def load_pretrained_backbone(model, ckpt_path, strict=False):
    """Load pre-trained backbone weights from a checkpoint.
    
    This function loads backbone weights from a pre-trained checkpoint and loads them
    into the model's BackboneTransformer. Handles both Lightning checkpoint format
    (with 'state_dict' key) and raw state_dict formats. Removes 'backbone.' prefix
    from keys if present.
    
    Parameters
    ----------
    model : BackboneAachenClassificationLightning
        The model to load weights into
    ckpt_path : str
        Path to the checkpoint file
    strict : bool, optional
        Whether to strictly enforce that the keys in state_dict match (default: False)
        When False, allows partial loading with dimension mismatches
    
    Returns
    -------
    dict
        Dictionary with 'missing_keys' and 'unexpected_keys' from the load operation
    """
    print(f"Loading pre-trained backbone weights from: {ckpt_path}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract state_dict from checkpoint
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        # Lightning checkpoint format
        state_dict = ckpt['state_dict']
    else:
        # Raw state_dict or direct model state
        state_dict = ckpt
    
    # Extract backbone state_dict and remove 'backbone.' prefix if present
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # Remove 'backbone.' prefix from Lightning checkpoint
            new_key = key[len('backbone.'):]
            backbone_state_dict[new_key] = value
        elif not any(key.startswith(prefix) for prefix in ['class_head', 'head', 'embedding_projection', 
                                                             'encoder_layer', 'cross_attn', 'fusion_proj']):
            # Include keys that don't belong to the classification head
            backbone_state_dict[key] = value
    
    if not backbone_state_dict:
        # If no backbone-specific keys found, try using all keys (for pure backbone checkpoints)
        backbone_state_dict = state_dict
        print("Warning: No 'backbone.' prefix found. Using all keys from checkpoint.")
    
    # Load state_dict into model.backbone
    result = model.backbone.load_state_dict(backbone_state_dict, strict=strict)
    missing_keys = result[0] if isinstance(result, tuple) else []
    unexpected_keys = result[1] if isinstance(result, tuple) else []
    
    if missing_keys:
        msg = f"Missing keys in backbone checkpoint: {missing_keys}"
        if strict:
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(msg)
    
    if unexpected_keys:
        msg = f"Unexpected keys in backbone checkpoint: {unexpected_keys}"
        logger.warning(msg)
    
    print(f"Successfully loaded backbone weights from: {ckpt_path}")
    
    return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}


def load_backbone_weights(model, ckpt_path, strict=False):
    """Wrapper function to load backbone weights.
    
    This is called from BackboneAachenClassificationLightning.on_train_start() hook.
    
    Parameters
    ----------
    model : BackboneAachenClassificationLightning
        The Lightning module to load weights into
    ckpt_path : str
        Path to the checkpoint file
    strict : bool, optional
        Whether to strictly enforce that the keys match (default: False)
    """
    if ckpt_path is None or ckpt_path == "None":
        logger.info("No pre-trained backbone weights specified.")
        return
    
    load_pretrained_backbone(model, ckpt_path, strict=strict)


def set_backbone_requires_grad(model, requires_grad=True):
    """Set requires_grad for all backbone parameters.
    
    Parameters
    ----------
    model : BackboneAachenClassificationLightning
        The Lightning module containing the backbone
    requires_grad : bool
        Whether to compute gradients for backbone parameters (default: True)
    """
    num_params = 0
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad
        num_params += param.numel()
    
    status = "trainable" if requires_grad else "frozen"
    print(f"Backbone set to {status}: {num_params:,} parameters")
    return num_params


def main():
    parser = argparse.ArgumentParser(description="OmniJet-alpha Anomaly Detection Training Script")
    parser.add_argument("--dataset_path", default=str(os.getenv("DATASET_PATH")), type=str, help="Path to the LHCO dataset")
    parser.add_argument("--gpu_id", type=int, default=int(os.getenv("GPU_ID")), help="GPU ID to use for computation")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED")), help="Random seed for reproducibility")
    parser.add_argument("--jet_name", type=str, default=str(os.getenv("JET_NAME")), choices=["jet1", "jet2", "both"], help="Name of the jet to use from the dataset")
    parser.add_argument("--merge_strategy", type=str, default=str(os.getenv("MERGE_STRATEGY")), choices=["concat", "average", "weighted_sum", "attention"], help="Merge strategy for dijet model")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE")), help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("MAX_STEPS")), help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE")), help="Learning rate")
    parser.add_argument("--train_val_split", type=float, default=float(os.getenv("TRAIN_VAL_SPLIT")), help="Train/validation split ratio")
    parser.add_argument("--n_jets_train", type=list, default=list(map(int,os.getenv("N_JETS_TRAIN").strip('[]').split(','))), help="Number of jets per class for training [signal, background]")
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM")), help="Embedding dimension")
    parser.add_argument("--naming_identifier", type=str, default="", help="Optional identifier to add to the run name for easier tracking")
    parser.add_argument("--pretrained_ckpt", type=str, help="Path to pre-trained checkpoint")
    parser.add_argument("--load_pretrained", action="store_true", help="Load pre-trained backbone weights from checkpoint")
    parser.add_argument("--log_dir", type=str, default=str(os.getenv("LOG_DIR_AACHEN")), help="Directory for experiment logs")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == 'true', default=True, help="Use automatic class weighting for imbalanced data (default: True)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights during training (no gradient updates)")
    parser.add_argument("--update_backbone", action="store_true", default=True, help="Update backbone weights during training (default: True, set to False with --freeze_backbone)")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=str(os.getenv("WANDB_PROJECT")), help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional, auto-generated if not provided)")
    
    # HPC arguments
    parser.add_argument("--use_hpc", action="store_true", help="Disable progress bar for HPC environments to avoid large log files")
    
    args = parser.parse_args()

    # ============================================================
    # 0. Initialize Experiment Logger
    # ============================================================
    exp_logger = ExperimentLogger(log_dir=args.log_dir, naming_identifier=args.naming_identifier)
    print(f"Experiment: {exp_logger.run_name}")
    print(f"Log directory: {exp_logger.run_dir}")
    
    # ============================================================
    # 1. Configuration
    # ============================================================
    # Set random seed for reproducibility
    L.seed_everything(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================
    # 2. Load Data
    # ============================================================

    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }

    signal_path = os.path.join(args.dataset_path, "sn_25k_SR_train.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    
    h5_files_all = [signal_path, background_path]

    print("n_jets_train:", args.n_jets_train)
    print("Using Jet:", args.jet_name)
    print("Merge strategy:", args.merge_strategy)


    # Log data configuration
    data_config = {
        "dataset_path": args.dataset_path,
        "signal_file": signal_path,
        "background_file": background_path,
        "n_jets_train": args.n_jets_train,
        "batch_size": args.batch_size,
        "max_sequence_len": 128,
        "mom4_format": "epxpypz",
        "train_val_split": args.train_val_split,
        "features": list(input_features_dict.keys()),
        "feature_preprocessing": input_features_dict,
        "shuffle_train": True,
        "jet_name": args.jet_name,
    }
    
    train_loader, val_loader = create_lhco_h5_dataloaders(
        h5_files_train=h5_files_all,
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=args.batch_size,
        n_jets_train=args.n_jets_train,  # [signal, background]
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        train_val_split=args.train_val_split,
        shuffle_train=True,
        num_workers=1,
    )

    # ============================================================
    # 3. Create Model
    # ============================================================

    # Calculate class weights for imbalanced dataset
    model_kwargs = create_model_config(input_features_dict, args)
    
    if args.use_class_weights:
        # n_jets_train = [signal, background] = [25000, 200000]
        n_signal = args.n_jets_train[0]
        n_background = args.n_jets_train[1]
        total = n_signal + n_background
        
        # Weight = total / (n_classes * n_samples_per_class)
        # Higher weight for minority class (signal)
        # Note: In LHCO files, signal=1, background=0
        weight_background = total / (2.0 * n_background)  # Weight for class 0
        weight_signal = total / (2.0 * n_signal)  # Weight for class 1
        # PyTorch CrossEntropyLoss expects weights in class order: [weight_for_class_0, weight_for_class_1]
        class_weights = [weight_background, weight_signal]
        
        print(f"\n=== Supervised Training Label Distribution ===")
        print(f"Class 0 (Background): {n_background} jets → weight={weight_background:.4f}")
        print(f"Class 1 (Signal): {n_signal} jets → weight={weight_signal:.4f}")
        print(f"Weight ratio (Signal/Background): {weight_signal/weight_background:.4f}")
        print(f"Class weights array: {class_weights}\n")
        model_kwargs["class_weights"] = class_weights
    else:
        print("Class weighting disabled - using standard CrossEntropyLoss")
        model_kwargs["class_weights"] = None

    # For constant learning rate, use ConstantLR
    scheduler_with_params = torch.optim.lr_scheduler.ConstantLR
    
    # For cosine annealing, uncomment the following:
    # scheduler_with_params = partial(
    #     torch.optim.lr_scheduler.CosineAnnealingLR,
    #     T_max=args.max_steps,
    #     eta_min=1e-6, # minimum learning rate
    # )

    # -------------------------------------------------------------------------
    # ---------------------- Single Jet Data Model ----------------------------
    # -------------------------------------------------------------------------

    # Initialize the Backbone + Classification Head (Single Jet)
    # model = BackboneClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": args.learning_rate,
    #         "weight_decay": 1e-2,
    #     },
    #     scheduler=scheduler_with_params,
    #     class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_argos",
    #         "mode": "max",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    # -------------------------------------------------------------------------
    # ------------------------- DiJet Data Model ------------------------------
    # -------------------------------------------------------------------------

    # model = BackboneDijetClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": args.learning_rate,
    #         "weight_decay": 1e-2,
    #     },
    #     scheduler=scheduler_with_params,
    #     merge_strategy=args.merge_strategy,  # other options: "average", "weighted_sum", "attention"
    #     class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_argos",
    #         "mode": "max",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    # -------------------------------------------------------------------------
    # ------------------ Aachen Anomaly Detection Model -----------------------
    # -------------------------------------------------------------------------

    model = BackboneAachenClassificationLightning(
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": args.learning_rate,
            "weight_decay": 1e-2,
        },
        scheduler=scheduler_with_params,
        merge_strategy=args.merge_strategy,  # other options: "average", "weighted_sum", "attention"
        model_kwargs=model_kwargs,
        use_continuous_input=True,
        scheduler_lightning_kwargs={
            "monitor": "val_argos",
            "mode": "max",
            "interval": "step",
            "frequency": 1,
        },
    )

    # Load pre-trained backbone weights if requested
    if args.load_pretrained and args.pretrained_ckpt:
        print(f"Loading pre-trained backbone weights from: {args.pretrained_ckpt}")
        load_pretrained_backbone(model, args.pretrained_ckpt)
        print("Successfully loaded pre-trained backbone weights!")
    
    # Freeze or update backbone based on arguments
    if args.freeze_backbone:
        set_backbone_requires_grad(model, requires_grad=False)
        print("Backbone frozen - only classification head will be trained")
    else:
        set_backbone_requires_grad(model, requires_grad=True)
        print("Backbone trainable - all parameters will be updated during training")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Log model configuration
    model_config = {
        "architecture": "BackboneAachenClassificationLightning",
        "merge_strategy": args.merge_strategy,
        "class_head_type": "aachen_attention",
        "use_continuous_input": True,
        "num_parameters": num_params,
        "embedding_dim": args.embedding_dim,
        "n_transformer_blocks": 2,
        "num_attention_heads": 8,
        "max_sequence_len": 128,
        "n_output_classes": 2,
        "freeze_backbone": args.freeze_backbone,
        "load_pretrained_checkpoint": args.load_pretrained and args.pretrained_ckpt,
        "pretrained_checkpoint_path": args.pretrained_ckpt if (args.load_pretrained and args.pretrained_ckpt) else None,
        "model_kwargs": {k: v for k, v in model_kwargs.items() if k != "particle_features_dict"},
    }
    
    # Log training configuration
    training_config = {
        "optimizer": "AdamW",
        "optimizer_params": {
            "lr": args.learning_rate,
            "weight_decay": 1e-2,
        },
        "scheduler": "ConstantLR",
        "max_steps": args.max_steps,
        "gradient_clip_val": 1.0,
        "precision": "32",
        "early_stopping_patience": 15,
        "early_stopping_monitor": "val_argos",
        "checkpoint_monitor": "val_argos",
        "checkpoint_mode": "max",
        "use_class_weights": args.use_class_weights,
        "class_weights": model_kwargs.get("class_weights", None),
    }
    
    # Log system configuration
    system_config = {
        "device": str(device),
        "gpu_id": args.gpu_id,
        "random_seed": args.seed,
        "timestamp_start": datetime.now().isoformat(),
    }
    
    # Combine all configs and log
    full_config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "system": system_config,
    }
    exp_logger.log_config(full_config)
    print(f"Configuration saved to: {exp_logger.run_dir / 'config.json'}")
    
    # Setup callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=exp_logger.get_checkpoint_dir(),
    #     filename="{epoch:02d}_{val_loss:.4f}",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=False,
    # )

    # Alternatively, monitor AUC instead of loss
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=exp_logger.get_checkpoint_dir(),
    #     filename="{epoch:02d}_{val_loss:.4f}",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=False,
    # )

    # Or using ARGOS metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_logger.get_checkpoint_dir(),
        filename="{epoch:02d}_{val_argos:.4f}",
        monitor="val_argos",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    # Early stopping enabled
    early_stop_callback = EarlyStopping(
        monitor="val_argos",
        patience=30,
        mode="max",
    )

    # AUC callback: computes ROC AUC on validation set each epoch and logs it
    auc_callback = AUCCallback()

    # ARGOS callback: computes ARGOS metric on validation set each epoch and logs it
    argos_callback = ARGOSCallback()

    # Setup loggers
    loggers = []
    
    # W&B logger
    if args.use_wandb:
        # Use custom run name if provided, otherwise use ExperimentLogger's run name
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_logger.run_name
        
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            save_dir=str(exp_logger.run_dir),
            config=full_config,
            log_model=True,  # Log model checkpoints to W&B
        )
        loggers.append(wandb_logger)
        
        print(f"\n{'=' * 80}")
        print(f"W&B logging enabled!")
        print(f"  Project: {args.wandb_project}")
        if args.wandb_entity:
            print(f"  Entity: {args.wandb_entity}")
        print(f"  Run name: {wandb_run_name}")
        print(f"  Run URL: {wandb_logger.experiment.url}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\nW&B logging disabled. Use --use_wandb to enable.\n")
        wandb_logger = None

    # Create trainer
    print("Starting training...")
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_callback, auc_callback, argos_callback, early_stop_callback],
        log_every_n_steps=20,
        val_check_interval=0.1,  # Validate every 10% of training data
        gradient_clip_val=1.0,
        precision="32",
        enable_progress_bar=not args.use_hpc,
        num_nodes=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ============================================================
    # 4. Training Loop
    # ============================================================
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        # Save final metrics and plots
        exp_logger.save_metrics()
        exp_logger.plot_training_curves()
        
        # Log final results
        exp_logger.log_final_results(trainer, checkpoint_callback)
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        print(f"Results saved to: {exp_logger.run_dir}")
        print(f"Training curves: {exp_logger.run_dir / 'plots' / 'training_curves.png'}")
        if args.use_wandb:
            print(f"W&B run: {wandb.run.url}")
            wandb.finish()
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user!")
        print(f"Partial results saved to: {exp_logger.run_dir}")
        print("=" * 80 + "\n")
        
        # Still try to save what we have
        try:
            exp_logger.save_metrics()
            exp_logger.plot_training_curves()
        except:
            pass
        
        # Finish W&B run
        if args.use_wandb:
            wandb.finish()
        
    except Exception as e:
        # Log error if training fails
        error_info = {
            "training_completed": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp_error": datetime.now().isoformat(),
        }
        exp_logger.log_results(error_info)
        print(f"\n{'=' * 80}")
        print(f"Training failed! Error logged to: {exp_logger.run_dir}")
        print(f"Error: {e}")
        print("=" * 80 + "\n")
        
        # Finish W&B run with error
        if args.use_wandb:
            wandb.finish(exit_code=1)
        
        raise


if __name__ == "__main__":
    main()