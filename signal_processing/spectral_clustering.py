import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyamg
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score

def fit_spectral_model(n_clusters, features):
    np.random.seed(42) # Setting seed as suggested by sklearn documentation when using amg solver

    model = SpectralClustering(n_clusters = n_clusters, eigen_solver='amg', affinity = 'rbf')
    labels = model.fit_predict(features)

    s_score = silhouette_score(features, labels)

    return labels, s_score

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "combined_sample.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')
    
    X = df.drop(columns='adversarial', axis=1)
    X = pca_transform_features(n_comp=2, features=X)
    y = df['adversarial']

    print('fitting spectral clustering model...')
    labels, score = fit_spectral_model(n_clusters=2, features=X)

    X['predicted clusters'] = labels
    X['true labels (adversarial = 1)'] = y

    acc_score = accuracy_score(y, labels)

    ax = sns.scatterplot(data = X, x=0, y=1, hue = 'predicted clusters', style='true labels (adversarial = 1)')
    ax.set(xlabel = 'Component 0',
           ylabel = 'Component 1',
           title = f'Spectral Clustering with 2 Components\nSilhouette Score: {round(score, 4)}\nAccuracy Score: {round(acc_score, 4):.2%}')
    plt.show()

if __name__:
    main()