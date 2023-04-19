# reference_protein_project
Code for work in "Adjusting for CSF Reference Proteins Improves Biomarker Accuracy", created by Linda Karlsson 2023.

This repository aims to provide details of how the results in "Adjusting for CSF Reference Proteins Improves Biomarker Accuracy" were created. Due to data sharing policies, most scripts and functions have been reduced to algorithmic descriptiveness and cannot be run without firstly specifying input data and data variable names. 

## Dependencies
Code was written in Python version 3.9 and R version 4.2. Required packages version:

Python:
- sklearn 0.0
- pandas 1.4.4
- numpy 1.23.3
- matplotlib 3.5.3
- pingouin 0.5.3
- tqdm 4.64.1
- statsmodels 0.13.2

R:
- tidyverse 1.3.2
- pROC 1.18.0


## File description

All created python functions were included in Functions.py, and examples of how they were used are given in the notebooks. In TSNE-clustering.ipynb, dimensionality reduction, K-means clustering and cluster evaluation is included. In Evaluation.ipynb, logistic regression evaluations and significance test on bootstrapped models can be found. Additionally, computation of linear regression and partial correlations are provided. In Robustness_Analysis, 
the methodology of evaluating models with added noise is given. 

The R-script Evaluation_testdata.R evaluates performance and significance on unseen testdata, comparing ROC curves. 
