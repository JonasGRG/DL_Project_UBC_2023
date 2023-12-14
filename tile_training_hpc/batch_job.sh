#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- specify GPU memory --
#BSUB -R "select[gpu80gb]"
### -- set the job Name --
#BSUB -J UBC_tile_sweep
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/train_%J.out
#BSUB -e batch_output/train_%J.err
# -- end of LSF options --

# Run the Python script (assuming train_sweep.py is in the same directory)
python train_sweep.py
