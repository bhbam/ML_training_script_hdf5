#!/bin/bash
#SBATCH -n 1                                        # use 1 cores
#SBATCH -c 50                                     # 16 tasks per core(threads)
#SBATCH -p gpu                                      # Submit to GPU partition
#SBATCH --qos gpu                                   # Submit to GPU QoS
#SBATCH --gres=gpu:1                                # Allocate GPU
#SBATCH -N 1                                        # only use 1 node — in this case, 16 cores on 1 node
#SBATCH --mem-per-cpu=5G                           # reserve 6G of memory
#SBATCH -J M_3p7_test               # job name
#SBATCH -o M_3p7_test.%A
#SBATCH --mail-user=bbbam@crimson.ua.edu

source /share/apps/modulefiles/conda_init.sh
conda activate Pytorch_VEN
module load compilers/intel/fakeintel
date
python  mass_inference_with_ieta_iphi_separete_train_valid_set.py UA_mass_config.ymal
date
