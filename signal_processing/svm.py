
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report   

def train_test_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel = 'rbf', C=1000, gamma=1, decision_function_shape='ovo')
    # model = SVC(kernel = 'rbf', C=1000, gamma=0.1, decision_function_shape='ovo')
    model.fit(X_train, y_train)
    class_label_predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, class_label_predictions)
    return model, acc_score, class_label_predictions

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features 

def main():

    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "multi_combined_sample_normalized.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Create Train and Test Samples
    X = df.drop(columns = 'adversarial', axis=1)
    y = df['adversarial']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=1234)

    # #Hyper parameter tuning
    # param_grid = {'C': [10, 100, 1000], 
    #             'gamma': [1, 0.1, 0.01],
    #             'decision_function_shape': ['ovo', 'ovr']}  
    # grid =  GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)

    model, acc_score, class_label_predictions = train_test_svm(X_train, X_test, y_train, y_test)
    print(classification_report(y_test, class_label_predictions))

    

main()