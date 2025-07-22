
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score, classification_report   

def train_test_svm(X_train, X_test, y_train, y_test, C=1000, gamma=1):
    model = SVC(kernel = 'rbf', C=1000, gamma=1, decision_function_shape='ovo')
    # model = SVC(kernel = 'rbf', C=1000, gamma=0.1, decision_function_shape='ovo')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, predictions)
    return acc_score, predictions

def train_test_OCSVM(X_train, X_test, y_train, y_test):
    model = OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.01)
    model.fit(X_train)
    predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, predictions)
    return model, acc_score, predictions

def pca_transform_features(n_comp, features):
    pca = PCA(n_components=n_comp)
    transformed_features = pd.DataFrame(pca.fit_transform(features))
    return transformed_features 

def svm_classifier(file):
    combined_df = pd.read_csv(file)
    combined_df = combined_df.drop(columns='Unnamed: 0')  # Ensure this column is dropped if it exists

    # Split into features and label
    X = combined_df.drop(columns='adversarial', axis=1)
    y = combined_df['adversarial']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    # Train and evaluate SVM model
    acc_score, label_predictions = train_test_svm(X_train, X_test, y_train, y_test) #, grid.best_params_['C'], grid.best_params_['gamma'])
    
    # Collect metrics
    precision = classification_report(y_test, label_predictions, output_dict=True)['1']['precision']
    recall = classification_report(y_test, label_predictions, output_dict=True)['1']['recall']
    f1_score = classification_report(y_test, label_predictions, output_dict=True)['1']['f1-score']
    acc_score = accuracy_score(y_test, label_predictions)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': acc_score
    }

def main():

    # Read in data 
    filename = os.path.join(os.getcwd(), "combined_obs_data_5000.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Create Train and Test Samples
    X = df.drop(columns = 'adversarial', axis=1)
    y = df['adversarial']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=1234)

    # #Hyper parameter tuning
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [0.0001, 0.001, 0.01, 0.1,1],
    #               'decision_function_shape': ['ovo', 'ovr']}
    # grid =  GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # grid.fit(X_train,y_train)
    # print(grid.best_params_)

    # Train and Evaluate OCSVM 
    # model, acc_score, class_label_predictions = train_test_OCSVM(X_train, X_test, y_train, y_test)
    # print(f"Accuracy Score: {acc_score}")
    # print(classification_report(y_test, class_label_predictions))

    acc_score, label_predictions = train_test_svm(X_train, X_test, y_train, y_test)
    print(classification_report(y_test, label_predictions))
    print(f"Accuracy Score: {acc_score}")
