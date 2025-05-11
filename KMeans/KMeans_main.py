'''
========================================================================
Creates correlation Biplot for feature analysis and implements three different methods to select required no of principal components
The results of selected no of PC are written into a file -> SelectPrincipalComponents.txt and Biplot is saved -> plot_PCA.png
Creates feature extracted dataset by reducing feature dimensions from N to select required number of principal components; file -> fatigue_Extracteddataset.csv
Creates elbow plot using PC1 and PC2 of extracted features. This plot is then used to select the number of centroids for clusters; plot -> plot_ElbowMethod.png
Creates clusters using the PC1 and PC2 of extracted features based on the selected number of centroids
Creates K-Means cluster plot -> plot_KmeanClustering.png and writes K-means result into the file -> resultKMeans.txt
Creates Parallel Coordinates plot to analyse the created clusters -> plot_parallelCoordinates and writes k-means summary into the file -> summaryKMeans.txt
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_KMeans import Data_preprocessing
from PPP_KMeans import Feature_extraction
from PPP_KMeans import Elbow_method
from PPP_KMeans import K_means

# Import required libraries
# -------------------------
import sys
import os
import click
import timeit

def check_inputs(k,kfind,selected_pc):
    '''
    ========================================================================
    Description:
    To check whether certain user inputs are within the required limits.
    ------------------------------------------------------------------------
    Parameters:
    k: Selected no of centroids based on the plot -> plot_ElbowMethod.png; dtype -> int
    kfind: It is the end limit of the list of k values starting from 1 to kfind from which appropriate no of centroids is identified for clustering; dtype -> int
    selected_pc: Selected no of PC based on the file -> SelectPrincipalComponents.txt; dtype -> int
    ------------------------------------------------------------------------
    Note:
    -If the inputs are not within the options range, program exits by throwing an error with possible inputs
    ========================================================================
    '''
    # Checking whether the input is correct or wrong
    #------------------------------------------------------------------------
    inputset, optionset = [k,kfind], ['k','kfind']                          # Grouping similar inputs and their options togather
    for idx,input in enumerate(inputset):
        if input <= 0:
            sys.exit('Error: '+str(optionset[idx])+' input must be > 0')    # If the input condition is not satisfied the program exits by throwing an error
    if selected_pc < 2:
        sys.exit('Error: selected_pc input must be >= 2')                   # Checking selected no of PC input
#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
@click.command()
@click.option('--data',nargs=1,type=str,default='fatigue_dataset.csv',help='Enter input dataset.csv: last column must be the target feature')
@click.option('--selected_pc',nargs=1,type=int,default=2,help='Select no of PC for extraction from: SelectPrincipalComponents.txt')
@click.option('--k',nargs=1,type=int,default=3,help='Select no of clusters from: plot_ElbowMethod.png')
@click.option('--kfind',nargs=1,type=int,default=9,help='Enter end limit for no of clusters in elbow plot')
@click.option('--target_column',nargs=1,type=str,default='Fatigue',help='Enter Target column header from dataset')
@click.option('--plt_size',nargs=4,type=float,default=([7.5, 5,7.5, 5]),help='Enter plot size for plot_Scree & plot_PCA')
#
# ============================================================================================================================================
#                                                      CREATING CLUSTERS --> K-MEANS CLUSTERING
# ============================================================================================================================================
#
def Kmeans_Clustering(data,selected_pc,k,kfind,target_column,plt_size):
    '''
    ========================================================================
    Description:
    Creating Principal components based on selected method and perfroming PCA analysis using Biplot. Creating clusters based on the number of centroids selected w.r.t 
    the elbow method plot and performing cluster analysis using parallel Coordinates.
    ========================================================================
    '''
    # Check certain user inputs
    #------------------------------------------------------------------------
    check_inputs(k,kfind,selected_pc)

    # Initialization for KMeans
    #------------------------------------------------------------------------
    Preprocessed_data = Data_preprocessing()  # Feature scaling
    extracting_X = Feature_extraction()       # Feature extraction
    n_clusters = Elbow_method()               # Elbow method to determine optimal number of clusters
    clusters = K_means()                      # Kmean clustering to make clusters with optimal clusters found from Elbow method

    # Data_preprocessing
    #------------------------------------------------------------------------
    start_time = timeit.default_timer()                                         # Start timer to note the time taken by K-Means to execute
    X,y = Preprocessed_data.import_dataset(data)                                # X: Independent feature, y: dependent feature
    scaled_X,scaled_y = Preprocessed_data.feature_scaling(X,y)                  # Scaled features

    # Principal Component Analysis(PCA) 
    #------------------------------------------------------------------------
    extracted_X,sort_eigen_val = extracting_X.PCA(scaled_X,selected_pc)         # Creating extracted features and initializing methods to select no of PC

    # Principal Component Analysis(PCA) method results 
    #------------------------------------------------------------------------
    extracting_X.PCA_results(plt_size)                                          # Writing methods results, eigen values and eigen vector; Creating scree plot

    # Plot to analyse extracted features
    #------------------------------------------------------------------------
    extracting_X.plot_PCA(extracted_X,X,plt_size)                               # Ploting Biplot for analysis

    # write extracted features
    #------------------------------------------------------------------------
    extracting_X.write_extractedFeatures(extracted_X,scaled_y,target_column)    # Writing the extracted features into a dataset

    # Plot to determine no of clusters 
    #------------------------------------------------------------------------
    n_clusters.plot_elbow_method(extracted_X[:,0:2],kfind)                      # Plot elbow method with first two PC

    # K-Means Clustering 
    #------------------------------------------------------------------------
    dataset,Centroids = clusters.make_clusters(extracted_X[:,0:2],k)            # Returns cluster results and centroid values

    # Plot K-Means Clustering and parallel coordinates for analysis 
    #------------------------------------------------------------------------
    clusters.plot_Kmean_clustering(k,dataset,Centroids,scaled_X,sort_eigen_val) # Plot clusters and parallel coordinates 

    # Write results
    #------------------------------------------------------------------------
    clusters.write_results(dataset,Centroids)                                   # Write k-means results and summary of clusters
    timetaken_bymod = timeit.default_timer()-start_time                         # Stop timer. Time taken by K-Means to execute is noted

    # Write time taken
    #------------------------------------------------------------------------
    save_path = os.path.abspath('Results_KMeans') # Save the file to the created directory
    with open(os.path.join(save_path, 'summaryKMeans.txt'), 'a') as f:
         print('\nTime taken by KMeans to execute(seconds):',timetaken_bymod,file=f) # Write time taken by K-Means to execute

if __name__ == '__main__':
    Kmeans_Clustering()

