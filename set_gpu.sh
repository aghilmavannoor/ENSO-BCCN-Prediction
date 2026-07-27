#!/bin/bash

# Purge all modules to avoid conflicts
module purge

module load cuda/12.6

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate ml_gpu

export CUDA_HOME=/home/apps/cuda-12.6
export CUDA_PATH=/home/apps/cuda-12.6
export LD_LIBRARY_PATH=/home/apps/cuda-12.6/lib64:/usr/lib64:/usr/local/cuda/lib64
export PATH=/home/apps/cuda-12.6/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/apps/cuda-12.6"

# Optional: print environment verification
echo "CUDA module loaded:"
module list

echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"

echo "Python:"
which python
