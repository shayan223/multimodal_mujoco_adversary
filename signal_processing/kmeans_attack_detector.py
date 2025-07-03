import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns
import matplotlib.pyplot as plt

# Take n random samples from each dataframe, join them, and standardize the data
def sample_and_combine(n_samples):
    #Read in Adversarial Dataset
    adv_file = os.path.join(os.getcwd(), "signal_processing", "data", "adversarial_obs_data.csv")
    adversarial_df = pd.read_csv(adv_file, header=0)

    #Read in Benign Dataset
    ben_file = os.path.join(os.getcwd(), "signal_processing", "data", "benign_obs_data.csv")
    benign_df = pd.read_csv(ben_file, header=0)

    # Drop the 'Unamed 0' Column 
    adversarial_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
    benign_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)

    # Add Labels
    adversarial_df['adversarial'] = 1
    benign_df['adversarial'] = 0

    # Sample Observations and Create Joined Dataframe
    adversarial_sampled = adversarial_df.sample(n=n_samples, random_state=4321)
    benign_sampled = benign_df.sample(n=n_samples, random_state=2341)
    combined_df = pd.concat([adversarial_sampled,benign_sampled])

    return combined_df

# Fit model, return silhouette score, labels, centroids
def fit_kmeans(df):
    model = KMeans(n_clusters = 2, random_state = 123)
    assigned_clusters = model.fit_predict(df)
    centroids = model.cluster_centers_
    
    s_score = silhouette_score(df,assigned_clusters) 
    
    return s_score, assigned_clusters, centroids

def pca_transform(df, n_components):
    pca = PCA(n_components=n_components)
    transformed_data = pd.DataFrame(pca.fit_transform(df))
    return transformed_data

def LDA_transform(df, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    transformed_data = pd.DataFrame(lda.fit_transform(df.drop(columns=['adversarial'], axis = 1), df['adversarial']))
    return transformed_data


def main():
    combined_df = sample_and_combine(5000)
    print(f'Combined Dataframe Shape: {combined_df.shape}')

    # Reduce to 2 components using PCA
    n_components = 2
    pca_transformed_data = pca_transform(combined_df.drop(columns=['adversarial']), n_components=n_components)
    
    # Fit Initial KMeans Model -- No Transformations
    s_score, clusters, centroids = fit_kmeans(pca_transformed_data)
    print(f'silhouette score ({n_components} components): {s_score}')
    
    # Visualize -- 2d
    pca_transformed_data['clusters'] = clusters
    ax = sns.scatterplot(data=pca_transformed_data, x=0, y=1, hue='clusters')
    sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color = 'k')
    ax.set(xlabel='Component 0', 
           ylabel='Component 1',
           title = f'KMeans Clustering with PCA (2 Components) \nSilhouette Score: {round(s_score, 4)}')
    plt.show()

    # Reduce to 3 components using PCA
    n_components = 3
    pca_transformed_data = pca_transform(combined_df.drop(columns=['adversarial']), n_components=n_components)
    
    # Fit Initial KMeans Model -- No Transformations
    s_score, clusters, centroids = fit_kmeans(pca_transformed_data)
    print(f'silhouette score ({n_components} components): {s_score}')

    # Visualize -- 3d
    pca_transformed_data['clusters'] = clusters
    x1 = pca_transformed_data[0]
    x2 = pca_transformed_data[1]
    x3 = pca_transformed_data[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster separately with a label
    for cluster_id in np.unique(clusters):
        idx = clusters == cluster_id
        ax.scatter(
            x1[idx], x2[idx], x3[idx],
            marker='o', s=5,
            label=f'Cluster {cluster_id}'
        )

    # Add centroids
    ax.scatter(
        centroids[:,0], centroids[:,1], centroids[:,2],
        color='k', marker='x', s=100, label='Centroids'
    )
    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')
    ax.set_title(f'KMeans Clustering with PCA ({n_components} Components)\nSilhouette Score: {round(s_score, 4)}')
    ax.legend()
    plt.show()

    # Fit LDA Model
    LDA_transformed_data = LDA_transform(combined_df, n_components=1) 
    s_score, clusters, centroids = fit_kmeans(LDA_transformed_data)
    print(f'silhouette score (LDA): {s_score}')

    LDA_transformed_data['clusters'] = clusters
    y_vals = [0] * LDA_transformed_data.shape[0]
    ax = sns.scatterplot(data=LDA_transformed_data, x=0, y=y_vals, hue='clusters')
    plt.scatter(x=centroids[:,0], y=[[0],[0]], color = 'k')
    ax.set(title = f'KMeans Clustering with LDA (1 Components) \nSilhouette Score: {round(s_score, 4)}')
    plt.show()


if __name__ == "__main__":
    main()  