#!/bin/sh
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -o script.out
#PBS -e script.err

# Load your required modules

module load anaconda3/personal
module load cuda
source ~/.bashrc
source activate /rds/general/user/hf721/home/anaconda3/envs/rna_env

export WORKDIR=$HOME/RNA_Kinetics/ModelsPerformances
echo $WORKDIR
cd $WORKDIR

python $WORKDIR/PerfModels.py

