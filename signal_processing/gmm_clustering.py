import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
import numpy as np


def fit_gmm(n_comp, features):
    gmm = GaussianMixture(n_components=n_comp, init_params='k-means++', random_state = 12)
    gmm.fit(features)

    labels = gmm.predict(features)

    s_score = silhouette_score(features, labels) 
    
    return labels, s_score

def relabel_clusters_by_majority(y_true, cluster_labels):
    # Create a DataFrame for easy grouping
    df = pd.DataFrame({'true': y_true, 'cluster': cluster_labels})
    mapping = {}
    for cluster in np.unique(cluster_labels):
        # Find the majority true label in this cluster
        majority_label = df[df['cluster'] == cluster]['true'].mode()[0]
        mapping[cluster] = majority_label
    # Map each cluster label to its majority true label
    relabeled = [mapping[c] for c in cluster_labels]
    return np.array(relabeled)

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def lda_transform_features(n_comp, features, label):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    transformed_features = pd.DataFrame(lda.fit_transform(features, label))
    return transformed_features


def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "combined_sample.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')
    
    # X = pd.DataFrame(preprocessing.normalize(df.drop(columns='adversarial', axis=1)))
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    X = pca_transform_features(n_comp=2, features=X)

    labels, score = fit_gmm(n_comp=2, features=X)
    
    X['assigned clusters'] = labels
    X['true labels (adversarial = 1)'] = y

    acc_score = accuracy_score(y, labels)

    ax = sns.scatterplot(data = X, x=0, y=1, hue = 'assigned clusters', style='true labels (adversarial = 1)')
    ax.set(xlabel = 'Component 0',
           ylabel = 'Component 1',
           title = f'GMM Clustering with 2 Components\nSilhouette Score: {round(score, 4)}\nAccuracy Score: {round(acc_score, 4):.2%}')
    plt.show()
if __name__:
    main()
