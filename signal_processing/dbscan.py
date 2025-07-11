import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "combined_sample_2.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')
    
    X = df.drop(columns='adversarial', axis=1)  
    # X = pd.DataFrame(MinMaxScaler().fit_transform(X))
    X = pca_transform_features(n_comp=3, features=X)
    y = df['adversarial']

    labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(X)

    s_score = silhouette_score(X, labels)
    acc_score = accuracy_score(y, labels)

    print(s_score)
    print(acc_score)

main()