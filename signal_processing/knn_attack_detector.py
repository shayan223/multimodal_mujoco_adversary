import pandas as pd
import os

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Train KNN model, return accuracy score
def train_test_knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
    model.fit(X_train, y_train)
    class_label_predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, class_label_predictions)
    return acc_score

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features

def main():
    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "combined_sample_2.csv")
    combined_df = pd.read_csv(filename)
    combined_df = combined_df.drop(columns='Unnamed: 0')

    # Split into features and label
    X = combined_df.drop(columns = 'adversarial', axis=1)
    y = combined_df['adversarial']

    # Feature Transformations
    # Normalization
    #X = pd.DataFrame(preprocessing.normalize(X), columns=X.columns)  # Normalize the data
    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X)) # MinMax Normalize the Data

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=1234)
    print("Created Train and Test Samples and Normalized Features.")

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

    # Use PCA to reduce to n components
    print(f"Applying PCA to reduce to {n_components} components...")
    n_components = 3     
    X_train = pca_transform_features(n_comp=n_components, features= X_train)
    X_test = pca_transform_features(n_comp=n_components, features= X_test)

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