'''
========================================================================
Test_file: test_KMeans_Clustering.py
Test_type: Testing with sample dataset
Aim: To check the correctness of implementation the model is tested with known dataset and verified whether it is able to recreate the expected clusters
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_KMeans import Data_preprocessing
from PPP_KMeans import Elbow_method
from PPP_KMeans import K_means

# Import required libraries
# -------------------------
import sys
import os
import click
import timeit
import numpy as np

def check_inputs(k,kfind):
    '''
    ========================================================================
    Description:
    To check whether certain user inputs are within the required limits.
    ------------------------------------------------------------------------
    Parameters:
    k: Selected no of centroids based on the plot -> plot_ElbowMethod.png; dtype -> int
    kfind: It is the end limit of the list of k values starting from 1 to kfind from which appropriate no of centroids is identified for clustering; dtype -> int
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
#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
def test_Kmeans_Clustering():
    @click.command()
    @click.option('--data',nargs=1,type=str,default='sample-data1_kmeans.csv',help='Enter input dataset.csv: last column must be the target feature')
    @click.option('--k',nargs=1,type=int,default=5,help='Select no of clusters from: plot_ElbowMethod.png')
    @click.option('--kfind',nargs=1,type=int,default=9,help='Enter end limit for no of clusters in elbow plot')

    #
    # ============================================================================================================================================
    #                                                      Validating --> K-MEANS CLUSTERING
    # ============================================================================================================================================
    #
    def Kmeans_Clustering(data,k,kfind):
        '''
        ========================================================================
        Description:
        Test for K-means Clustering; Check whether the obtained clusters in 'plot_KmeanClustering.png' from the directory test_results is same as 
        the expected clusters from the directory expected_results.
        ========================================================================
        '''
        # Check certain user inputs
        #------------------------------------------------------------------------
        check_inputs(k,kfind)

        # Initialization for KMeans
        #------------------------------------------------------------------------
        Preprocessed_data = Data_preprocessing()  # Feature scaling
        n_clusters = Elbow_method()               # Elbow method to determine optimal number of clusters
        clusters = K_means()                      # Kmean clustering to make clusters with optimal clusters found from Elbow method

        # Data_preprocessing
        #------------------------------------------------------------------------
        start_time = timeit.default_timer()                                         # Start timer to note the time taken by K-Means to execute
        X = Preprocessed_data.import_dataset(data)                                  # X: Independent feature
        scaled_X = Preprocessed_data.feature_scaling(X).values                      # Scaled features

        # Plot to determine no of clusters 
        #------------------------------------------------------------------------
        n_clusters.plot_elbow_method(scaled_X,kfind)                                # Plot elbow method

        # K-Means Clustering 
        #------------------------------------------------------------------------
        dataset,Centroids = clusters.make_clusters(scaled_X,k)                      # Returns cluster results and centroid values

        # Plot K-Means Clustering and parallel coordinates for analysis 
        #------------------------------------------------------------------------
        clusters.plot_Kmean_clustering(k,dataset,Centroids)                         # Plot clusters 

        # Write results
        #------------------------------------------------------------------------
        clusters.write_results(dataset,Centroids)                                   # Write k-means results and summary of clusters
        timetaken_bymod = timeit.default_timer()-start_time                         # Stop timer. Time taken by K-Means to execute is noted

        # Write time taken
        #------------------------------------------------------------------------
        save_path = os.path.abspath('test_results') # Save the file to the created directory
        with open(os.path.join(save_path, 'summaryKMeans.txt'), 'a') as f:
            print('\nTime taken by KMeans to execute(seconds):',timetaken_bymod,file=f) # Write time taken by K-Means to execute
        
        # Check whether the mean of the standardized data = 0 (close enough)
        #------------------------------------------------------------------------
        np.testing.assert_almost_equal(np.mean(scaled_X),0)
        
        # Check whether the standard deviation of the standardized data = 1 (close enough)
        #------------------------------------------------------------------------
        np.testing.assert_almost_equal(np.std(scaled_X),1)

        
    if __name__ == '__main__':
        Kmeans_Clustering()

test_Kmeans_Clustering()
