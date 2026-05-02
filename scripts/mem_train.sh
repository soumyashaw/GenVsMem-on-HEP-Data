#!/bin/bash
# Wrapper called by HTCondor/SLURM for one chunk of memorization trials.
# Usage: mem_train.sh <chunk_id> [extra args...]
set -e

CHUNK_ID=$1
shift  # remaining args forwarded to the Python script

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"
PROJECT_DIR="$HOME/GenVsMem-on-HEP-Data"

echo "=========================================="
echo "Host       : $(hostname)"
echo "Chunk ID   : $CHUNK_ID"
echo "Started at : $(date)"
echo "=========================================="
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"

apptainer exec --nv "$IMG" bash -c "
    source /opt/conda/bin/activate
    cd \"$PROJECT_DIR\"
    python -m src.train.train_mem \
        --chunk_id $CHUNK_ID \
        --chunk_size 200 \
        --n_trials 4000 \
        $@
"

echo "Chunk $CHUNK_ID finished at $(date)"
