# KMeans and KNN as Adversarial Attack Classifiers
Fit KMeans and K nearest neigbors to agent observation data to see if traditional signal processing techniques perform better detecting adversarial attacks.

## Required Packages
- Pandas 
- Numpy
- Matplotlib
- Seaborn
- scikit-learn

## Included Files
- **kmeans_attack_detector.py** : Implements 3 KMeans models. Two with PCA applied to the data before fitting and one with LDA applied before fitting. 
- **Knn_attack_detector.py**: Implements 2 KNN Models. One with no transformations applied, and one with PCA applied. 
- **join_datasets.py**: Script used to label and join original adversarial and benign datasets. Joined data set saved to data/combined_labeled_obs_data.csv.
