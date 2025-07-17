
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score, classification_report   

def train_test_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel = 'rbf', C=1000, gamma=1, decision_function_shape='ovo')
    # model = SVC(kernel = 'rbf', C=1000, gamma=0.1, decision_function_shape='ovo')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, predictions)
    return model, acc_score, predictions

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

def main():

    # Read in data 
    filename = os.path.join(os.getcwd(), "signal_processing", "data", "mutli_fgsm015_combined_sample.csv")
    df = pd.read_csv(filename)
    df = df.drop(columns='Unnamed: 0')

    # Create Train and Test Samples
    X = df.drop(columns = 'adversarial', axis=1)
    y = df['adversarial']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=1234)

    # #Hyper parameter tuning
    # param_grid = {'degree': [1,2,3],
    #             'gamma': [1, 0.1, 0.01],
    #             'nu': [0.001, 0.01, 0.1]}  
    # grid =  GridSearchCV(OneClassSVM(), param_grid, scoring= 'accuracy' , refit=True, verbose=3)
    # grid.fit(X_train,y_train)
    # print(grid.best_params_)

    model, acc_score, class_label_predictions = train_test_svm(X_train, X_test, y_train, y_test)
    print(classification_report(y_test, class_label_predictions))

    # model, acc_score, class_label_predictions = train_test_OCSVM(X_train, X_test, y_train, y_test)
    # print(f"Accuracy Score: {acc_score}")
    # print(classification_report(y_test, class_label_predictions))


    

main()