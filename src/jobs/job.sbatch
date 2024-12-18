#!/bin/bash
#SBATCH --job-name=GloVe_NN
#SBATCH --output=output_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
python3 glove_template.py