
# Protein Language Model

## Overview

This repository contains a modular implementation of a protein language model using ESM2-embeddings, TensorFlow, and a JAX/FLAX-based architecture. It is an extension of the Protein Language Model found in Deep Learning for Biology and uses the CAFA 3 dataset. 
## Improvements 

Feature Engineering

To improve model generalisation, the training dataset was augmented with protein sequences from species with >80% homology to humans (Mus musculus and Rattus norvegicus). This helps the model distinguish protein functions across mammals more effectively and significantly increases the dataset size.

Mitigating Overfitting

I implemented MMseqs2 to cluster sequences into ~500 functional groups. By applying Group Stratified K-Fold cross-validation, I reduced overfitting to ensure the model is evaluated on non-overlapping clusters, which prevents data leakage from similar sequences and provides a more accurate assessment of accuracy than train/test/split.

Architecture Tuning

MLP had additional layers added to improve generalisation and better capture sequence-function relationships. 
## Comparison to original model 

Compared to the baseline, the recall (true positive rate) is almost doubled, whilst maintaining a higher level of precision (ratio of true positives to all predicted positives). AUROC and accuracy are misleading due to the imbalance multilabel setting.


### My Model

| split | loss | accuracy | recall | precision | auprc | auroc |
|-------|----------|-----------|----------|-----------|---------|---------|
| valid | 0.045695 | 0.987719 | 0.238473 | 0.495350 | 0.471522| 0.928473|
| test | 0.049019 | 0.986966 | 0.221798 | 0.493831 | 0.455864| 0.923144|

### Original Model

| split | loss | accuracy | recall | precision | auprc | auroc |
|-------|----------|-----------|----------|-----------|---------|---------|
| valid | 0.080156 | 0.978457 | 0.126869 | 0.418515 | 0.411870| 0.880883|
| test | 0.080675 | 0.978032 | 0.125820 | 0.435193 | 0.410439| 0.879234| 


Analysis

Models converge within the first 50 steps, with validation metrics closely following training performance. AUPRC is still steadily improving, so it may benefit from additional steps or additional architectures to improve recall further. Precision is improving faster than recall, suggesting the model is confident but still missing out on true predictions.  

!(assets/Loss_Validation_Metrics.png)

Predictions generally capture broad trends but miss out on rarer functions. The model still overpredicts functions of certain families of proteins, which are visible as faint lines. 

!(assets/Functional_annotations.png)

The model outperforms simple coin flips or proportional predictions.

!(assets/Best_performing.png)

## Repository structure 
Python 3.10

Protein-Language-Model/ 
├── assets/ # Visualisations
├── notebooks/ # Polished demo notebook 
├── src/ # Key functions  
├── .gitignore # Shields repo from large .feather/data/embedding files 
├── ReadMe.md
├── requirements-lock.txt # Exhaustive dependencies (Python 3.10 + multiple dependencies)
└── requirements.txt # Core dependencies 

Clone the Repo

git clone https://github.com/josephWGLee/Protein-Language-Model.git
cd Protein-Language-Model 

Install Dependencies 

pip install -r requirements.txt 


Note on Data: Raw embedding files are excluded due to size constraints. However, scripts to download the CAFA 3 datasets and the ESM-2 weights are included within the notebooks/ directory. 

## Future Directions

Due to the computational intensity of full token-level embeddings, this version utilises mean-pooled sequence representations. I attempted to do this using a regular laptop, but I did not have the memory or capacity to make this realistically feasible. Future work could include token-level embeddings, 1D-CNN layers, training the ESM2 model,  and migrating the TensorFlow functions to PyTorch.
