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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def lda_transform_features(n_comp, features, label):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    transformed_features = pd.DataFrame(lda.fit_transform(features, label))
    return transformed_features
    

def best_spectral(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    models = []

    # Spectral Clustering on 2D PCA
    X_pca_2 = pca_transform_features(n_comp=2, features=X)
    labels_2, score_2 = fit_spectral_model(n_clusters=2, features=X_pca_2)
    acc_score_2 = accuracy_score(y, labels_2)
    models.append({
        'model': 'Spectral Clustering (2D PCA)',
        'silhouette_score': score_2,
        'accuracy': acc_score_2,
        'labels': labels_2
    })

    # Spectral Clustering on 3D PCA
    X_pca_3 = pca_transform_features(n_comp=3, features=X)
    labels_3, score_3 = fit_spectral_model(n_clusters=2, features=X_pca_3)
    acc_score_3 = accuracy_score(y, labels_3)
    models.append({
        'model': 'Spectral Clustering (3D PCA)',
        'silhouette_score': score_3,
        'accuracy': acc_score_3,
        'labels': labels_3
    })

    # Spectral Clustering on LDA
    X_lda = lda_transform_features(n_comp=1, features=X, label=y)
    labels_lda, score_lda = fit_spectral_model(n_clusters=2, features=X_lda)
    acc_score_lda = accuracy_score(y, labels_lda)
    models.append({
        'model': 'Spectral Clustering (LDA)',
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

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "combined_obs_data_5000.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')
    
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    # Fits unreduced model
    # labels, score = fit_spectral_model(n_clusters=2, features=X)
    # acc_score = accuracy_score(y, labels)
    # print(f's-score: {score}, acc score: {acc_score}')

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

# if __name__:
#     main()