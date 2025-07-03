import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

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

# Train KNN model, return accuracy score
def train_test_knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
    model.fit(X_train, y_train)
    class_label_predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, class_label_predictions)
    return acc_score

def pca_transform(df, n_components):
    pca = PCA(n_components=n_components)
    transformed_data = pd.DataFrame(pca.fit_transform(df))
    return transformed_data

def main():
    combined_df = sample_and_combine(5000)

    # Create Train and Test Samples
    X = combined_df.drop(columns = 'adversarial', axis=1)
    y = combined_df['adversarial']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=1234)

    # Train and evaluate for k values 1-20
    print("Training KNN Classifier with k values from 1 to 20...")
    acc_scores = []
    k_values = range(1,22)
    for i in k_values:
        score = train_test_knn(X_train, X_test, y_train, y_test,i)
        acc_scores.append(score)

    # Find optimal k value
    optimal_k = acc_scores.index(max(acc_scores))+1  # k value with highest accuracy score
    print(f'Optimal k value: {optimal_k} with accuracy score: {max(acc_scores)}')

    # Fit model with optimal k value and return classification report
    print("Fitting KNN Classifier with optimal k value...")
    model = KNeighborsClassifier(n_neighbors = optimal_k, algorithm = 'kd_tree')
    model.fit(X_train, y_train)
    label_predictions = model.predict(X_test)

    print(classification_report(y_test, label_predictions))

    # Use PCA to reduce to 2 components
    n_components = 2       
    X_train = pca_transform(X_train, n_components=n_components)
    X_test = pca_transform(X_test, n_components=n_components)

    # Train and evaluate for k values 1-20
    print("Training KNN Classifier with k values from 1 to 20...")
    acc_scores = []
    k_values = range(1,22)
    for i in k_values:
        score = train_test_knn(X_train, X_test, y_train, y_test,i)
        acc_scores.append(score)
    
    optimal_k = acc_scores.index(max(acc_scores))+1  # k value with highest accuracy score
    print(f'Optimal k value: {optimal_k} with accuracy score: {max(acc_scores)}')

    # Fit model with optimal k value and return classification report
    print("Fitting KNN Classifier with optimal k value...")
    model = KNeighborsClassifier(n_neighbors = optimal_k, algorithm = 'kd_tree')
    model.fit(X_train, y_train)     
    label_predictions = model.predict(X_test)   
    print(classification_report(y_test, label_predictions)) 

if __name__ == "__main__":
    main()