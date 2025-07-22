from knn_attack_detector import *
from kmeans_attack_detector import *
from svm import *
from gmm_clustering import *
from spectral_clustering import *
import os

filename = os.path.join(os.getcwd(), "signal_processing/dataset_samples/combined_velocity_fgsm015_data_5000_normalized.csv")
# Set up results dictionaries
classifier_results = {'Models': [], 'Accuracy': [], 'f1 score': [], 'precision': [], 'recall': []}

clustering_results = {'Models': [], 'Silhouette_score': [], 'Clustering Accuracy': []}

def run_all_models():
    # Start with classification models
    print("Running KNN Attack Detector...")
    knn_results = knn_classifer(filename)

    precision = knn_results.pop('precision')
    recall = knn_results.pop('recall')
    f1_score = knn_results.pop('f1_score') 
    accuracy = knn_results.pop('accuracy')

    classifier_results['Models'].append(f'KNN (k={knn_results["optimal_k"]})')
    classifier_results['Accuracy'].append(accuracy)
    classifier_results['f1 score'].append(f1_score)
    classifier_results['precision'].append(precision)  
    classifier_results['recall'].append(recall)      
    print(f"KNN Results recorded.") 

    print("Running SVM Attack Detector...")
    svm_results = svm_classifier(filename)

    precision = svm_results.pop('precision')
    recall = svm_results.pop('recall')
    f1_score = svm_results.pop('f1_score')
    accuracy = svm_results.pop('accuracy')

    classifier_results['Models'].append('SVM')
    classifier_results['Accuracy'].append(accuracy)
    classifier_results['f1 score'].append(f1_score)
    classifier_results['precision'].append(precision)
    classifier_results['recall'].append(recall)
    print(f"SVM Results recorded.")


    # Clustering models...
    print("Running KMeans Clustering...")
    kmeans_results = best_kmeans(filename)
    silhouette_score = kmeans_results.pop('silhouette_score')
    acc_score = kmeans_results.pop('accuracy')  

    clustering_results['Models'].append(kmeans_results['model'])
    clustering_results['Silhouette_score'].append(silhouette_score)
    clustering_results['Clustering Accuracy'].append(acc_score)
    print(f"KMeans Results recorded.")

    print("Running GMM Clustering...")
    gmm_results = best_gmm(filename)
    silhouette_score = gmm_results.pop('silhouette_score')
    acc_score = gmm_results.pop('accuracy') 

    clustering_results['Models'].append(gmm_results['model'])
    clustering_results['Silhouette_score'].append(silhouette_score)
    clustering_results['Clustering Accuracy'].append(acc_score)
    print(f"GMM Results recorded.")

    print("Running Spectral Clustering...")
    spectral_results = best_spectral(filename)
    silhouette_score = spectral_results.pop('silhouette_score')
    acc_score = spectral_results.pop('accuracy')   
    clustering_results['Models'].append(spectral_results['model'])
    clustering_results['Silhouette_score'].append(silhouette_score)
    clustering_results['Clustering Accuracy'].append(acc_score)
    print(f"Spectral Clustering Results recorded.")


def check_results():
    print("\nResults Summary:")
    print("Classifier Results:")
    for i in range(len(classifier_results['Models'])):
        print(f"{classifier_results['Models'][i]}: Accuracy={classifier_results['Accuracy'][i]}, "
              f"f1 score={classifier_results['f1 score'][i]}, "
              f"precision={classifier_results['precision'][i]}, "
              f"recall={classifier_results['recall'][i]}")
        
    print("\nClustering Results:")
    for i in range(len(clustering_results['Models'])):
        print(f"{clustering_results['Models'][i]}: "
              f"Silhouette Score={clustering_results['Silhouette_score'][i]}, "
              f"Clustering Accuracy={clustering_results['Clustering Accuracy'][i]}")    
    print("\n")

def save_results():
    # Create new directory for dataset if it doesn't exist
    directory_name = 'velocity_fsgm015'

    if not os.path.exists('signal_processing/best_models_results/' + directory_name):
        os.makedirs('signal_processing/best_models_results/' + directory_name)

    # Define the path to save results
    path = os.path.join(os.getcwd(), 'signal_processing', 'best_models_results', directory_name)

    # Save classifier results
    classifier_df = pd.DataFrame(classifier_results)
    classifier_df.to_csv(os.path.join(path,'normalized_data_classifier_results.csv'), index=False)

    # Save clustering results
    clustering_df = pd.DataFrame(clustering_results)
    clustering_df.to_csv(os.path.join(path, 'normalized_data_clustering_results.csv'), index=False)

    print(f"Results saved to 'normalized_data_classifier_results.csv' and 'normalized_data_clustering_results.csv' in signal_processing/best_models_results/{directory_name}/.")

def debug():
    df = pd.read_csv(filename)
    print(df.head())
if __name__ == "__main__":
    # debug()
    run_all_models()
    check_results()
    save_results()
    # You can add more model runs here as needed
    # For example, run_kmeans(), run_svm(), etc.
    # Each function should update the classifier_results or clustering_results dictionaries accordingly.

    