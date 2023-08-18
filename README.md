# BAP1 Texture Analysis Project

## Organization

- requirements.txt: Configuration file listing packages used for this project. Please update if you install additional packages.

### FeatureExtraction
This folder contains code for performing feature extraction on CT images and associated segmentations.
- extract.py: Script to call in order to run a feature extraction experiment.
- FeatureExtractor.py: Code for Feature Extractor class that performs feature extraction.
- perturbation.py: Code for image perturbations used in assessing feature robustness.
- configs/: Text files used to specify experiment configurations.
    - extraction.txt: Configuration for full feature extraction. 
    - perturbation.txt: Configuration for full feature extraction with image perturbations. 

### Classification
This folder contains code to perform BAP1 classification on extracted radiomic features.
- classify.py: Script to implement final classification pipeline for BAP1 mutation status differentiation.
- classifyRobust.py: Script to compare classification results on various robust feature sets.
- util.py: Code for various functions used in classification scripts (loading data, selecting features, etc.)
- crossValidation.py: Mena's code for cross validation of BAP1 classification pipeline.

### FeatureRobustness
This folder contains code for analyzing feature robustness using various metrics of agreement and image perturbations.
- getRobustnessMetrics.py: Script to generate robustness metrics from features before and after perturbation.
- getRobustFeatures.py: Script to generate robust feature subsets based on combinations of various metrics.
- metrics.py: Code for agreement metrics used to assess feature robustness.
- featureStability.py: Script to perform analysis of stability of feature selection on robust feature sets.
- plotsAndTests.py: Various plots and statistical tests for presenting results of feature robustness analysis.
- Notebooks/: Jupyter notebooks exploring individual methods of analyzing agreement.

### DatasetStatistics
This folder contains scripts to gather statistics about the BAP1 dataset and generate relevant plots.
- BAP1DatasetStatistics.ipynb: Notebook that gathers and organizes demographic and CT-related information about the BAP1 dataset.
- correlationHeatmap.ipynb: Notebook to generate heatmaps and clustered heatmaps of correlated features.
- dataExploration.py, variousPlots.py: Miscellaneous plots about the dataset and results.

### DataPreprocessing
This folder contains miscellaneous scripts to preprocess data before feature extraction and classification.
- createImageDir.py: Script to create initial data directory configured for feature extraction experiments.
- resampleSliceThickness.py: Script to generate axial CT images with resampled slice thickness to standardize future analysis.

### Figures + Findings
This folder contains summative figures and tables related to the BAP1 dataset.

## Setup
- Create a conda environment with Python 3.11
- Install packages: pip install -r requirements.txt
- There are issues installing Nyxus via pip. Also run: conda install -c conda-forge nyxus=0.6.0

## Running Feature Extraction
- Modify config file with experiment-specific parameters.
- Run: python extract.py --config configs/name_of_config.txt
