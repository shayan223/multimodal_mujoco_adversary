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
    
def best_gmm(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    # Fit GMM
    labels, score = fit_gmm(n_comp=2, features=X)

    # Relabel clusters by majority true label
    relabeled_labels = relabel_clusters_by_majority(y, labels)
    acc_score = accuracy_score(y, relabeled_labels)
    
    return {'silhouette_score': score, 'accuracy': acc_score}

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "combined_obs_data_5000.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = df.drop(columns='adversarial', axis=1)
    y = df['adversarial']

    labels, score = fit_gmm(n_comp=2, features=X)
    acc_score = accuracy_score(y, labels)
    print(f's-score: {score}, acc score: {acc_score}')

    # Reduce to 3 components with PCA
    X_pca_3 = pca_transform_features(n_comp=3, features=X)

    # Fit Model
    labels, score = fit_gmm(n_comp=2, features=X_pca_3)
    
    # Restore predicted and true labels
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
    ax.set_title(f'GMM Clustering with PCA (3 Components)\nSilhouette Score: {round(score, 4)} \nAccuracy Score: {round(acc_score, 4):.2%}')
    ax.legend()
    plt.show()

    # Reduce to 2 components with PCA
    X_pca_2 = pca_transform_features(n_comp=2, features=X)

    # Fit Model
    labels, score = fit_gmm(n_comp=2, features=X_pca_2)
    
    # Restore predicted and true labels
    X_pca_2['predicted clusters'] = labels
    X_pca_2['true labels (adversarial = 1)'] = y

    acc_score = accuracy_score(y, labels)


    # Visualize 2d
    ax = sns.scatterplot(data = X_pca_2, x=0, y=1, hue = 'predicted clusters', style='true labels (adversarial = 1)')
    ax.set(xlabel = 'Component 0',
           ylabel = 'Component 1',
           title = f'GMM Clustering with 2 Components\nSilhouette Score: {round(score, 4)}\nAccuracy Score: {round(acc_score, 4):.2%}')
    plt.show()

# if __name__:
#     main()
