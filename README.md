# Enhancing Next POI Recommendations with Cluster-Aware Graph Attention Networks

# Dataset
The Dataset is called Foursquare Dataset and can be downloaded from https://sites.google.com/site/yangdingqi/home/foursquare-dataset (section 2). It is composed by the check-in of Tokyo (TKY) and New York (NYC). In this project is used only the first one. The file is called "dataset_TSMC2014_TKY". More dataset details can be found in the dataset_TSMC2014_readme.txt file in the dataset structure.
## Preprocessing

To prepare the data, you have to to run the script `preprocess.py`.  
Put the script in the same folder as the dataset file.

```bash
python preprocess.py
```
The code saves 7 tensors in main folder:
- tensor_poi_coord
- tensor_poi_locations
- tensor_poi_categories
- check_in_tuples_history_cut
- validation_dataset
- test_dataset
- train_dataset

# Train
To train the model you need to run the script `train.py` and the folder must contains all the tensors generated from the preprocess.
```bash
python train.py
```
# Evaluate
To evaluate the model you need to run the script `evaluate.py` and the folder must contains all the tensors generated from the preprocess.
Before the evaluation of the model, using the script `evaluate.py`, you have to execute the file `train.py`.
```bash
python evaluate.py
```
