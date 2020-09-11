#!/bin/bash
#SBATCH -A SNIC2019-5-169
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:k80:1



# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
deactivate

# Load the module environment suitable for the job
#module load Python-bare/3.6.6
#module load GCC/8.2.0-2.31.1 OpenMPI/3.1.3 PyTorch/1.1.0-Python-3.7.2
module load GCC/8.2.0-2.31.1  CUDA/10.1.105  OpenMPI/3.1.3 torchvision/0.3.0-Python-3.7.2 scikit-learn/0.20.3 matplotlib/3.0.3-Python-3.7.2
source newThesisProjEnv/bin/activate

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"


python main.py --epochs 200 --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 224 --momentum 0.9 -val --evaluate --upsample --pretrained

echo "Program finished with exit code $? at: `date`"
