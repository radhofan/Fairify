#!/usr/bin/env bash

curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda <<< "Yes"

export PATH="$HOME/miniconda/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/miniconda"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -c conda-forge mamba -y

mamba shell init --shell=bash
source ~/.bashrc  
eval "$(mamba shell hook --shell=bash)"

mamba create -n fairify python=3.9 -y
source $HOME/miniconda/bin/activate fairify
mamba activate fairify

pip install -r Fairify/requirements.txt
pip install tqdm
sudo apt install csvtool
sudo apt install -y python3-swiftclient
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Retrain existing model with counterexamples + synthethic data
# python Fairify/src/AC/new_model_2.py

python Fairify/src/AC/train_fair_model.py
python Fairify/src/AC/metric_aif360.py
python Fairify/src/AC/metric_themis_causality.py
python Fairify/src/AC/metric_random_unfairness.py
# bash Fairify/src/fairify.sh Fairify/src/AC/Verify-AC-experiment-v2.py

# python Fairify/src/BM/train_fair_model.py
# python Fairify/src/BM/metric_aif360.py
# python Fairify/src/BM/metric_themis_causality.py
# python Fairify/src/BM/metric_random_unfairness.py
# bash Fairify/src/fairify.sh Fairify/src/BM/Verify-BM-experiment-v2.py

# python Fairify/src/GC/train_fair_model.py
# python Fairify/src/GC/metric_aif360.py
# python Fairify/src/GC/metric_themis_causality.py
# python Fairify/src/GC/metric_random_unfairness.py
# bash Fairify/src/fairify.sh Fairify/src/GC/Verify-GC-experiment-v2.py

# Run fairify experiment
# bash Fairify/src/fairify.sh Fairify/src/GC/Verify-GC-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/AC/Verify-AC-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/BM/Verify-BM-experiment.py
# bash Fairify/src/fairify.sh Fairify/src/CP/Verify-CP.py
# bash Fairify/src/fairify.sh Fairify/src/DF/Verify-DF.py

# source ~/openrc

# bucket_name="bare_metal_experiment_pattern_data" 
# file_to_upload="Fairify/src/GC/res/counterexample.csv"

# echo
# echo "Uploading results to the object store container $bucket_name"
# swift post $bucket_name

# if [ -f "$file_to_upload" ]; then
#     echo "Uploading $file_to_upload"
#     swift upload "$bucket_name" "$file_to_upload" --object-name "counterexample.csv"
# else
#     echo "ERROR: File $file_to_upload does not exist!" >&2
#     exit 1
# fi