# BAP1 Texture Analysis Project

## Organization

- requirements.txt: Configuration file listing packages used for this project. Please update if you install additional packages.

### FeatureExtraction
This folder contains code for performing feature extraction on CT images and associated segmentations.
- extract.py: Script to call in order to run a feature extraction experiment.
- FeatureExtractor.py: Code for Feature Extractor class that performs feature extraction.
- perturbation.py: Code for image perturbations used in assessing feature robustness.
- configs/: Text files used to specify experiment configurations.

### DataPreprocessing
This folder contains miscellaneous scripts to preprocess data before feature extraction and classification.
- createImageDir.py: Script to create initial data directory configured for feature extraction experiments.
- resampleSliceThickness.py: Script to generate axial CT images with resampled slice thickness to standardize future analysis.

## Setup
- Create a conda environment with Python 3.11
- Install packages: pip install -r requirements.txt
- There are issues installing Nyxus via pip. Also run: conda install -c conda-forge nyxus=0.6.0

## Running Feature Extraction

### To run feature extraction script: 
- Modify config file with experiment-specific parameters.
- Run: python extract.py --config configs/name_of_config.txt
