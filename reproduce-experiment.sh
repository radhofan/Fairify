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
sudo apt install -y python3-swiftclient

# Retrain existing model with counterexamples + synthethic data
# python Fairify/src/AC/new_model_2.py
# python Fairify/src/AC/metrics.py
# python Fairify/src/AC/metrics2.py
# python Fairify/src/AC/detect_bias.py

# Run fairify experiment
# bash Fairify/src/fairify.sh Fairify/src/GC/Verify-GC-experiment.py
bash Fairify/src/fairify.sh Fairify/src/GC/Verify-GC-experiment-new2.py
# bash Fairify/src/fairify.sh Fairify/src/AC/Verify-AC-experiment-new.py
# bash Fairify/src/fairify.sh Fairify/src/AC/Verify-AC-experiment-new2.py
# bash Fairify/src/fairify.sh Fairify/src/BM/Verify-BM-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/CP/Verify-CP.py
# bash Fairify/src/fairify.sh Fairify/src/DF/Verify-DF.py

source ~/openrc

bucket_name="bare_metal_experiment_pattern_data" 
file_to_upload="Fairify/src/GC/res-age/counterexample.csv"

echo
echo "Uploading results to the object store container $bucket_name"
swift post $bucket_name

if [ -f "$file_to_upload" ]; then
    echo "Uploading $file_to_upload"
    swift upload "$bucket_name" "$file_to_upload" --object-name "counterexamples.csv"
else
    echo "ERROR: File $file_to_upload does not exist!" >&2
    exit 1
fi