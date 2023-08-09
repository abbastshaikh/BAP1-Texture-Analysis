# BAP1 Texture Analysis Project

## Organization

### FeatureExtraction
This folder contains code for performing feature extraction on CT images and associated segmentations.
- extract.py: Script to call in order to run a feature extraction experiment.
- FeatureExtractor.py: Code for Feature Extractor class that performs feature extraction.
- perturbation.py: Code for image perturbations used in assessing feature robustness.
- configs/: Text files used to specify experiment configurations.

## Setup

### Package Installation:
Create a conda environment with Python 3.11: conda create -n nameOfEnvironment python=3.11 \
Install packages: pip install -r requirements.txt \
There are issues installing Nyxus via pip. Also run: conda install -c conda-forge nyxus=0.6.0

## Running Feature Extraction

### To run feature extraction script: 
- Modify config file with experiment-specific parameters.
- Run: python extract.py --config configs/name_of_config.txt
