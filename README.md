# BAP1-Texture-Analysis

## To setup environment and install packages:
Create a conda environment with Python 3.11: conda create -n nameOfEnvironment python=3.11 \
Install packages: pip install -r requirements.txt \
There are issues installing Nyxus via pip. Also run: conda install -c conda-forge nyxus=0.6.0

## To run feature extraction script: 
Modify config file with experiment-specific parameters. \
Run: python FeatureExtraction.py --config configs/name_of_config.txt
