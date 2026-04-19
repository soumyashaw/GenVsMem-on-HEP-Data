# GenVsMem-on-HEP-Data
Evalaution of Generalization vs Memorization on HEP Data

Instruction to activate interactive GPU:
condor_submit -interactive -append 'request_gpus=1' -append 'request_cpus=1' -append 'request_memory=16GB'
salloc --partition=c23g --time=01:00:00 --nodes=1 --cpus-per-task=2 --mem=16G --gres=gpu:1

Activating Apptainer Shell with GPU and Bind Mount:
apptainer exec --nv /.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif bash
source /opt/conda/bin/activate