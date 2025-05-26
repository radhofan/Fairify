#!/usr/bin/env bash

# Install Miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda <<< "Yes"

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/miniconda"

# Install Mamba
conda install -c conda-forge mamba -y

# Initialize Mamba shell
mamba shell init --shell=bash
source ~/.bashrc  # Reload shell config
eval "$(mamba shell hook --shell=bash)"

# Create and activate conda environment
mamba create -n fairify python=3.9 -y
source $HOME/miniconda/bin/activate fairify
mamba activate fairify

# install requirements
pip install -r Fairify/requirements.txt
pip install tqdm
sudo apt install csvtool

# Retrain existing model with counterexamples + synthethic data
# python Fairify/src/GC/new_model.py

# Run fairify experiment
# bash Fairify/src/fairify.sh Fairify/src/GC/Verify-GC-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/AC/Verify-AC-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/BM/Verify-BM-experiment.py
bash Fairify/src/fairify.sh Fairify/src/CP/Verify-CP.py
# bash Fairify/src/fairify.sh Fairify/src/DF/Verify-DF.py