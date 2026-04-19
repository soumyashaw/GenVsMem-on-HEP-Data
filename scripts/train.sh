#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/GenVsMem-on-HEP-Data

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    cd \"$HOME/GenVsMem-on-HEP-Data\"
    python -m src.train.train --jet_name both --merge_strategy concat --batch_size 64 --naming_identifier poc_catas_forget_b3_s1 --inject_freq 1 --num_signal_jets 1 --use_wandb
    "

echo "Job finished."
