import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Fit model, return silhouette score, labels, centroids
def fit_kmeans(df):
    model = KMeans(n_clusters = 2, random_state = 123)
    clusters = model.fit_predict(df)
    centroids = model.cluster_centers_
    
    s_score = silhouette_score(df, clusters) 
    
    return s_score,clusters,centroids

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def LDA_transform_features(n_comp, features, label):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    transformed_data = pd.DataFrame(lda.fit_transform(features,label))
    return transformed_data

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "combined_sample_2.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = df.drop(columns='adversarial')
    y = df['adversarial']

    # Feature transformations
    # Normalization Methods
    # X = pd.DataFrame(normalize(X))
    X = pd.DataFrame(MinMaxScaler().fit_transform(X))
    # PCA 
    # X = pca_transform_features(n_comp=3, features=X)
    # LDA 
    X = LDA_transform_features(n_comp=1, features=X, label=y)


    # Fit KMeans Model 
    s_score, clusters, centroids = fit_kmeans(X)
    print(f'Silhouette score: {s_score}')

    # Calculate clustering accuracy
    acc_score = accuracy_score(y, clusters)
    print(f'Accuracy Score: {acc_score:.2%}')

    # Add transformed features, true labels, and predicted labels to df
    plot_df = pd.concat([X,y], axis=1)
    plot_df['clusters'] = clusters

    # # Visualize -- 2d
    # ax = sns.scatterplot(data=plot_df, x=0, y=1, hue='clusters', style='adversarial')
    # sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color = 'k')
    # ax.set(xlabel='Component 0', 
    #        ylabel='Component 1',
    #        title = f'KMeans Clustering with PCA (2 Components) \nSilhouette Score: {round(s_score, 4)}\nAccuracy Score: {round(acc_score, 4):.2%}')
    # plt.show()

    # # Visualize -- 3d
    # x1 = plot_df[0]
    # x2 = plot_df[1]
    # x3 = plot_df[2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot each cluster separately with a label
    # for cluster_id in np.unique(clusters):
    #     idx = clusters == cluster_id
    #     ax.scatter(
    #         x1[idx], x2[idx], x3[idx],
    #         marker='o', s=5,
    #         label=f'Cluster {cluster_id}'
    #     )

    # # Add centroids
    # ax.scatter(
    #     centroids[:,0], centroids[:,1], centroids[:,2],
    #     color='k', marker='x', s=100, label='Centroids'
    # )
    # ax.set_xlabel('Component 0')
    # ax.set_ylabel('Component 1')
    # ax.set_zlabel('Component 2')
    # ax.set_title(f'KMeans Clustering with PCA (3 Components)\nSilhouette Score: {round(s_score, 4)} \nAccuracy Score: {round(acc_score, 4):.2%}')
    # ax.legend()
    # plt.show()

    # Visualize
    y_vals = [0] * plot_df.shape[0]
    ax = sns.scatterplot(data=plot_df, x=0, y=y_vals, hue='clusters')
    plt.scatter(x=centroids[:,0], y=[[0],[0]], color = 'k')
    ax.set(title = f'KMeans Clustering with LDA (1 Component)\nSilhouette Score: {round(s_score, 4)} \nAccuracy Score: {round(acc_score, 4):.2%}')
    plt.show()


if __name__ == "__main__":
    main()  