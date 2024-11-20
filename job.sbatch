#!/bin/bash
#SBATCH --job-name=GloVe_NN
#SBATCH --output=output_%j.log

python3 glove_template.py