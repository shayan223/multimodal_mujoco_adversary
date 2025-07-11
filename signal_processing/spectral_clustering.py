import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyamg
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler

def fit_spectral_model(n_clusters, features):
    # np.random.seed(42) # Setting seed as suggested by sklearn documentation when using amg solver

    model = SpectralClustering(n_clusters = n_clusters, eigen_solver='amg', random_state= 123, affinity = 'rbf')
    labels = model.fit_predict(features)

    s_score = silhouette_score(features, labels)

    return labels, s_score

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "multi_combined_sample.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')
    
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    labels, score = fit_spectral_model(n_clusters=2, features=X)
    acc_score = accuracy_score(y, labels)
    print(f's-score: {score}, acc score: {acc_score}')

    # PCA transform
    X_pca_3 = pca_transform_features(n_comp=3, features=X)

    print('fitting spectral clustering model...')
    labels, score = fit_spectral_model(n_clusters=2, features=X_pca_3)

    X_pca_3['predicted clusters'] = labels
    X_pca_3['true labels (adversarial = 1)'] = y

    acc_score = accuracy_score(y, labels)

    # Visualize -- 3d
    x1 = X_pca_3[0]
    x2 = X_pca_3[1]
    x3 = X_pca_3[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster separately with a label
    for cluster_id in np.unique(labels):
        idx = labels == cluster_id
        ax.scatter(
            x1[idx], x2[idx], x3[idx],
            marker='o', s=5,
            label=f'Cluster {cluster_id}'
        )

    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')
    ax.set_title(f'Spectral Clustering with PCA (3 Components)\nSilhouette Score: {round(score, 4)} \nAccuracy Score: {round(acc_score, 4):.2%}')
    ax.legend()
    plt.show()

    # PCA transform
    X_pca_2 = pca_transform_features(n_comp=2, features=X)
    y = df['adversarial']

    print('fitting spectral clustering model...')
    labels, score = fit_spectral_model(n_clusters=2, features=X_pca_2)

    X_pca_2['predicted clusters'] = labels
    X_pca_2['true labels (adversarial = 1)'] = y

    acc_score = accuracy_score(y, labels)

    ax = sns.scatterplot(data = X_pca_2, x=0, y=1, hue = 'predicted clusters', style='true labels (adversarial = 1)')
    ax.set(xlabel = 'Component 0',
           ylabel = 'Component 1',
           title = f'Spectral Clustering with 2 Components\nSilhouette Score: {round(score, 4)}\nAccuracy Score: {round(acc_score, 4):.2%}')
    plt.show()

if __name__:
    main()