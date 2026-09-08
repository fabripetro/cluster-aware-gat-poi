# Enhancing Next POI Recommendations with Cluster-Aware Graph Attention Networks

This repository contains the implementation of a **Cluster-Aware Graph Attention Network** for next Point-of-Interest (POI) category prediction on urban mobility datasets.

---

## Overview
Predicting user movement between POIs requires capturing both short-term spatial patterns and long-term spatio-temporal dependencies. This project introduces a novel framework combining:

- **Spatial Clustering**: K-Means clustering (K=6) on POI coordinates to separate local and inter-cluster dynamics.
- **Graph Neural Networks**: **GCN** for intra-cluster local flow and **GAT** for cross-cluster transitions.
- **Spatio-Temporal LSTM**: Sequence modeling incorporating time and spatial interval embeddings.
- **Cross-Attention Matching**: Candidate POI matching via cross-attention between trajectory representations and target candidate embeddings.

---

## Experimental Results
Evaluated on the **Foursquare Tokyo (TKY)** check-in dataset:

- **Recall@1**: **21.7%** (Outperforms baseline models such as STGCAN, STAN, and DeepMove by ~27%).
- **Recall@5**: **31.5%**
- **Recall@10**: **43.9%**

---

## Project Structure
- `project/`: Contains all Python modules (`preprocess.py`, `train.py`, `evaluate.py`, and neural network components).
- `report.pdf`: Full project report and documentation.
- `LICENSE`: Project open-source license.
- `README.md`: Main project documentation.

---

## Dataset
The dataset used is the **Foursquare Dataset** (Tokyo check-ins: `dataset_TSMC2014_TKY`).
- **Download link**: [Foursquare Dataset (Section 2)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- Details about dataset features are available in the `dataset_TSMC2014_readme.txt` file within the dataset archive.

---

## Getting Started & Usage

### 1. Preprocessing
To prepare the data, place `preprocess.py` in the same directory as the dataset file and run:

`python project/preprocess.py`

This script processes check-in histories, performs clustering, and generates 7 PyTorch tensors required for training and testing:
- `tensor_poi_coord`
- `tensor_poi_locations`
- `tensor_poi_categories`
- `check_in_tuples_history_cut`
- `validation_dataset`
- `test_dataset`
- `train_dataset`

### 2. Training
To train the model, run the training script ensuring all generated tensors are available in the working folder:

`python project/train.py`

### 3. Evaluation
To evaluate the trained model on the test dataset, execute `train.py` first, then run:

`python project/evaluate.py`

---

## Frameworks & Dependencies
- Python 3.x
- PyTorch / PyTorch Geometric
- Scikit-Learn
- Pandas
- NumPy

---
