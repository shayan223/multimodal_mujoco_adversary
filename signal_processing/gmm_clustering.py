import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np


def fit_gmm(n_comp, features):
    gmm = GaussianMixture(n_components=n_comp, init_params='k-means++', random_state = 12)
    gmm.fit(features)

    labels = gmm.predict(features)

    s_score = silhouette_score(features, labels) 
    
    return labels, s_score

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def lda_transform_features(n_comp, features, label):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    transformed_features = pd.DataFrame(lda.fit_transform(features, label))
    return transformed_features
    
def best_gmm(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    models = []

    # GMM on 2D PCA
    X_pca_2 = pca_transform_features(n_comp=2, features=X)
    labels_2, score_2 = fit_gmm(n_comp=2, features=X_pca_2)
    acc_score_2 = accuracy_score(y, labels_2)
    models.append({
        'model': 'GMM (2D PCA)',
        'silhouette_score': score_2,
        'accuracy': acc_score_2,
        'labels': labels_2
    })

    # GMM on 3D PCA
    X_pca_3 = pca_transform_features(n_comp=3, features=X)
    labels_3, score_3 = fit_gmm(n_comp=2, features=X_pca_3)
    acc_score_3 = accuracy_score(y, labels_3)
    models.append({
        'model': 'GMM (3D PCA)',
        'silhouette_score': score_3,
        'accuracy': acc_score_3,
        'labels': labels_3
    })

    # GMM on LDA
    X_lda = lda_transform_features(n_comp=1, features=X, label=y)
    labels_lda, score_lda = fit_gmm(n_comp=2, features=X_lda)
    acc_score_lda = accuracy_score(y, labels_lda)
    models.append({
        'model': 'GMM (LDA)',
        'silhouette_score': score_lda,
        'accuracy': acc_score_lda,
        'labels': labels_lda
    })

    # Select the model with the highest accuracy
    best_model = max(models, key=lambda m: m['accuracy'])

    # Return only relevant results
    return {
        'model': best_model['model'],
        'silhouette_score': best_model['silhouette_score'],
        'accuracy': best_model['accuracy'],
    }

if __name__ == "__main__":
    print(best_gmm(os.path.join(os.getcwd(), "signal_processing/dataset_samples/combined_angular_data_5000.csv")))
