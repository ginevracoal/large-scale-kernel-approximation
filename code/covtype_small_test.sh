#!/bin/bash

#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:kepler:2
#SBATCH --time=8:00:00
#SBATCH --partition=gll_usr_gpuprod
#SBATCH --account=uts18_bortoldl_0

. virtualenv_2/bin/activate
python /galileo/home/userexternal/gcarbone/individual/code/covtype_small_test.py
