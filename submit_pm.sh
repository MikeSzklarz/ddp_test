#!/bin/bash 

#SBATCH --job-name=ddp_test_%j                      # Job name
#SBATCH --output=logs/slurm/job_%j.txt              # Output log
#SBATCH --nodes=2                                   # Number of nodes
#SBATCH --ntasks-per-node=1                         # Number of tasks to invoke on each node                               # Number of GPUs per node
#SBATCH --mem=65536                                 # Memory (64 GB)
#SBATCH --time=30-00:00:00                          # Job time limit
#SBATCH --partition=waccamaw                        # Partition to use
#SBATCH --exclusive                                 # Exclusive node allocation
#SBATCH --exclude=waccamaw03,waccamaw04             # Exclude specific nodes

LOGDIR=${PWD}/logs
mkdir -p ${LOGDIR}
args="${@}"

source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate ddp_test

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher

export WORLD_SIZE=$SLURM_NTASKS  
export RANK=$SLURM_PROCID        
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DEISTRIBUTED_DEBUG=INFO

# Having multiple gpus available when only one is avaiable causes issues
export CUDA_VISIBLE_DEVICES=0

# printing all env variables

echo "MASTER_ADDR: $MASTER_ADDR"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $MASTER_PORT"

# Debugging mode
set -x

# Run the training script directly
srun torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_timeout=300 main.py
