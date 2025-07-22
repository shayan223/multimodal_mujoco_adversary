# Adversarial Attack Detection on Multimodal Reinforcement Learning (RL) Models
This project involves exploring different detection methods for FGSM attacks on soft actor-critic. 

## Required Packages
- Pandas 
- Numpy
- Matplotlib
- Seaborn
- scikit-learn

## Included Files
### Data Pre-Processing and Metric Collection
- **join_datasets.py**: Script used to label and join original adversarial and benign datasets. 
- **min_max_normalize_data.py**: Normalizes a given DataFrame.
- **run_all_models.py**: Runs all models for a given DataFrame and saves results to a specified destination. 
### Shallow Classifiers
- **svm.py**
- **knn_attack_detector.py**
### Clustering Methods
- **kmeans_attack_detector.py** 
- **gmm_clustering.py**
- **spectral_clustering.py**


