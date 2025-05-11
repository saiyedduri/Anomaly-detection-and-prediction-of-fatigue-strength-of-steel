'''
========================================================================
Machine learning algorithm: K-Means Clustering(unsupervised machine learning algorithm for Clustering)
========================================================================
'''
# Import required libraries
#------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#
# ======================================================================
# Preprocessing the dataset
# ======================================================================
#
class Data_preprocessing:
    '''
    ========================================================================
    Description:
    Data preprocessing is done to transform raw input data into a readable and understandable format.
    ======================================================================== 
    '''
    def import_dataset(self,data):
        '''
        ========================================================================
        Description:
        Reading in the input dataset and separating independent features.
        ------------------------------------------------------------------------
        Parameters:
        data: Input dataset with independent features and dependent feature
        ------------------------------------------------------------------------
        Conditions:
        The data file must be of the format .csv
        The data entries of the dataset should be real numbers and the empty onces should be filled based on the domain knowledge or zeros 
        There must exist atleast one independent feature and the dependent feature must be the last column of the dataset
        ------------------------------------------------------------------------
        Return:
        X: Dataframe of independent features; Size -> [No of samples x independent features]
        ======================================================================== 
        '''
        dataset = pd.read_csv(data)
        X = dataset.iloc[:,:-1]
        return X

    def feature_scaling(self,X):
        '''
        ========================================================================
        Description:
        Standardizing the independent features into a unifrom scale.
        ------------------------------------------------------------------------
        Parameters:
        X: Dataframe of independent features; Size -> [No of samples x independent features]
        ------------------------------------------------------------------------
        Check:
        The mean of the standardized data = 0 (close enough)
        The standard deviation of the standardized data = 1 (close enough)
        ------------------------------------------------------------------------
        Return:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        ========================================================================
        '''
        mean_X, std_X = np.array([np.mean(X,axis =0)]), np.array([np.std(X,axis =0)]) # Mean computed row wise for independent features(@axis = 0)
        scaled_X = (X - mean_X)/std_X                                                 
        return scaled_X
#
# ======================================================================
# Construction of Elbow method
# ======================================================================
#
class Elbow_method:
    '''
    ========================================================================
    Description:
    Elbow method -> to choose optimal number of clusters in K-means clustering.
    ======================================================================== 
    '''

    def calculate_WCSS(self,extracted_X,Clusters,Centroids):
        '''
        ========================================================================
        Description:
        Storing the WCSS value of each cluster.
        WCSS: within-clusters sum of squares(i.e variance).
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into two features; Size -> [No of samples x two independent features]
        Clusters: Datapoints belong to each centroids are clusters; Size -> [No of samples x 1] 
        Centroids: The mean value of each clusters; Size -> [No of centroids x two independent features]
        ------------------------------------------------------------------------
        Return:
        sum_distance: Calculating the sum of distance between the finial centroid and its clusters; dtype -> float    
        ------------------------------------------------------------------------
        Note:
        -The reduction of WCSS determines the optimal number of clusters required for a given dataset
        -No of clusters should be equivalent to no of centroids and vice versa
        ========================================================================
        '''
        sum_distance = 0                                                # Initially assigning sum to be zero
        for i,val in enumerate(extracted_X):

            sum_distance += np.sqrt((Centroids[int(Clusters[i]),0] - val[0])**2 + (Centroids[int(Clusters[i]),1] - val[1])**2)
            # Calculating the distance between the datapoint and centroid of the cluster it belongs
        return sum_distance

    def determine_clusters(self,extracted_X,k):
        '''
        ========================================================================
        Description:
        Determine the number of clusters based on the range of k values.
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into two features; Size -> [No of samples x two independent features]
        k: Number of centroids; dtype -> int
        ------------------------------------------------------------------------
        Return:
        sum_distance: Calculating the sum of distance between the finial centroid and its clusters; dtype -> float      
        ========================================================================
        '''                             
        extracted_Xdf = pd.DataFrame(extracted_X)                                           # Creating a dataframe of the extracted feature
        Clusters = np.zeros(extracted_X.shape[0])                                           # Creating storage for clusters
        Centroids = extracted_Xdf.sample(k,random_state = 0).values                         # Randomly selecting centroids based on k value
        check = 1                                                                           # Stop(check -> 0) when centroids and updated centroids are the same i.e. the sum
        while check:                                                                        # of the distance between the datapoints and its clusters centroid is at the minimum
            # Assign datapoints to the centroid
            for i , row in enumerate(extracted_X):                                          # Assigning cluster to each datapoint of extracted feature
                minimum_distance = float('inf')                                             # Setting minimum distance to infinity
                for index,Centroid in enumerate(Centroids):
                    distance = np.sqrt((Centroid[0]-row[0])**2 + (Centroid[1]-row[1])**2)   # Calculating the distance between each centroid and the datapoint

                    if minimum_distance > distance:                                         # Finding which cluster the datapoint belongs to
                        minimum_distance = distance
                        Clusters[i] = index                                                 # Clustering the datapoints based on the nearest centroids

            # Create updated centroids
            #------------------------------------------------------------------------
            Centroids_update =  extracted_Xdf.groupby(Clusters).mean().values               # Calculating the mean values of each clusters

            # Check wheather clusters are there for the centroid
            #------------------------------------------------------------------------
            if not len(Centroids) == (max(Clusters)+1):
                sys.exit('Retry: Clusters are only possible till k = %d. Update kfind to %d'%(k-1,k-1)) # If number of clusters are not equal to centroids through an error with possible k value

            if np.count_nonzero(Centroids-Centroids_update) ==0:                            # If satisfied exit the loop else update centroids and recalculate
                check = 0
            else:
                Centroids = Centroids_update

        sum_distance = n_clusters.calculate_WCSS(extracted_X,Clusters, Centroids)           # Calculating within-clusters sum of squares
        return sum_distance
    
    def plot_elbow_method(self,extracted_X,k):
        '''
        ========================================================================
        Description:
        Creating elbow plot to select the number of centroids.
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into two features; Size -> [No of samples x two independent features]
        k: Number of centroids; dtype -> int
        ------------------------------------------------------------------------
        Output:
        Creating a directory if one does not exists: test_results and returns a plot: plot_ElbowMethod.png      
        ========================================================================
        '''
        K = np.arange(1,k+1,1)          # Plotting for different number of centroids                                      
        WCSS = []                       # Creating storage for within-clusters sum of squares
        for value in K:
            sum_distance = n_clusters.determine_clusters(extracted_X,value) # Calculating within-clusters sum of squares
            WCSS.append(sum_distance)   # Storing the value of within-clusters sum of squares for each number of centroids

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('test_results'):
                os.makedirs('test_results')
        save_path = os.path.abspath('test_results') # Save the file to the created directory
        
        # Plot elbow method
        #------------------------------------------------------------------------
        plt.title('The Elbow method')               # Set plot title
        plt.xlabel('Number of clusters')            # Set label for x-axis -> Number of clusters
        plt.ylabel('WCSS')                          # Set label for y-axis -> WCSS
        plt.plot(K,WCSS,'o--',label = 'varience')   
        plt.grid(axis ='x')                         # Make grids in x axis
        plt.legend()
        fig = plt.gcf()                                                                 # Get the current figure
        fig.set_size_inches((7.5, 5),forward = False)                                   # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_ElbowMethod.png'),bbox_inches='tight') # Saving the plot
        plt.clf()                                                                       # Cleaning the current figure after saving
#
# ======================================================================
# Construction of K-Means clustering to make clusters
# ======================================================================
#
class K_means:
    '''
    ========================================================================
    Description:
    K_means -> making clusters and creating cluster plots.
    ======================================================================== 
    '''
    
    def make_clusters(self,extracted_X,k_clusters):
        '''
        ========================================================================
        Description:
        Making clusters of datapoints w.r.t the nearest centroids.
        Centroids are selected w.r.t the plot -> plot_ElbowMethod.png present in the directory: Plots_KMeans.
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into two features; Size -> [No of samples x two independent features]
        k_clusters: Number of centroids; dtype -> int
        ------------------------------------------------------------------------
        Return:
        extracted_X: Returns a dataframe of extracted features updated with additional columns of distance between the features and the centroids and the
        information of which cluster each feature belongs to; Size -> [No of samples x two independent features + no of centroids + 1(cluster details)]
        Centroids: The mean value of each clusters; Size -> [No of centroids x two independent features]      
        ========================================================================
        '''
        extracted_X = pd.DataFrame(extracted_X)                                     # Creating a dataframe of extracted features
        extracted_X= pd.DataFrame.rename(extracted_X,columns ={0:'PC1',1:'PC2'})    # Renaming the columns of the features into PC1 and PC2                                       
        Centroids = extracted_X.sample(k_clusters,random_state = 0)                 # In the N number of extracted features select K random features(coordinates) as centroids
        check = 1                                                                   # Stop(check -> 0) when centroids and updated centroids are the same i.e. the sum
        while check:                                                                # of the distance between the datapoints and its clusters centroid is at the minimum

            # Loading the distance between all the datapoints with each centroid into the data frame
            #------------------------------------------------------------------------
            index = 1                                                               # Column header for first centroid
            for i, centroid in Centroids.iterrows():                                # Iterate through the rows of the dataframe
                euclidean_distance = np.zeros(extracted_X.shape[0])                 # Creating storage to store the distance between the datapoints and each centroids
                for idx,datapoints in extracted_X.iterrows():
                    distance = np.sqrt(np.sum([(centroid['PC1'] - datapoints['PC1'])**2 , (centroid['PC2'] - datapoints['PC2'])**2]))
                    euclidean_distance[idx] = distance                              # Storing the calculated distance between  all the datapoints with each centroids     
                
                extracted_X[index] = euclidean_distance                             # Loading the stored calculated distance into the dataframe
                index += 1                                                          # Updating column header for each centroids

            # Creating clusters by selecting the minimum distance between the datapoints and the centroids 
            #------------------------------------------------------------------------
            cluster = np.zeros(extracted_X.shape[0])                                # Creating storage to store the cluster detail of each datapoints
            for j,row in extracted_X.iterrows():                                    # Finding the minimum distance between the centroids for each datapoints
                minimum_distance = row[1]                                           # Intially considering datapoints are closer to first centroid

                # Checking which centroid the datapoint belongs to
                #------------------------------------------------------------------------
                cluster_index = 1                                           
                for k in range(k_clusters):
                    if row[k+1] < minimum_distance:                                 # Finding which centroid the datapoint belongs to
                        minimum_distance = row[k+1]                                 # Among the centroids select centroid with minimum distance for each datapoint
                        cluster_index = k+1                                         # Assign the datapoint to the cluster based on the selected centroid 
                cluster[j] = cluster_index                                          # Storing cluster detail of each datapoints 
            extracted_X['Cluster'] = cluster                                        # Loading the cluster detail of each datapoints into the dataframe

            # Create updated centroids
            #-----------------------------------------------------------------------
            Centroids_update = extracted_X.groupby(['Cluster']).mean()[['PC1','PC2']] # Calculating the mean values of each clusters
            
            if np.count_nonzero(Centroids-Centroids_update) == 0:                   # If satisfied exit the loop else update centroids and recalculate
                check = 0
            else:
                Centroids = Centroids_update

        Clusters = extracted_X['Cluster'].values                                    # Cluster details of all datapoints
        Centroids = Centroids['PC1'], Centroids['PC2']                              # Data coordinates of the centroids
        for idx in range (k_clusters):
            extracted_X= pd.DataFrame.rename(extracted_X,columns ={idx+1:'Dist_Centroid%d'%(idx+1)}) # Renaming the header of distance between the centroids and datapoints
        return extracted_X,Centroids

    def plot_Kmean_clustering(self,k_clusters,extracted_X,Centroids):   
        '''
        ========================================================================
        Description:
        Make cluster plot from the created clusters.
        ------------------------------------------------------------------------
        Parameters:
        k_clusters: Number of centroids; dtype -> int
        extracted_X: Returns a dataframe of extracted features updated with additional columns of distance between the features and the centroids and the
        information of which cluster each feature belongs to; Size -> [No of samples x two independent features + no of centroids + 1(cluster details)]
        Centroids: The mean value of each clusters; Size -> [No of centroids x two independent features]
        ------------------------------------------------------------------------
        Output:
        Creating a directory if one does not exists: test_results and returns two plots: plot_KmeanClustering.png     
        ========================================================================
        '''   
        # Creating a directory if one does not exists
        #-----------------------------------------------------------------------
        if not os.path.exists('test_results'):
                os.makedirs('test_results')
        save_path = os.path.abspath('test_results') # Save the file to the created directory

        for k in range(k_clusters):
            dataset=extracted_X[extracted_X['Cluster']==k+1]      # To plot each cluster in different colour, each cluster will be ploted one after another
            plt.scatter(dataset['PC1'],dataset['PC2'],label='cluster %d'%(k+1))
        
        # Plot K-Means clustering     
        #------------------------------------------------------------------------
        plt.scatter(Centroids[0],Centroids[1], c='black',marker='*',s=150,label='centroids')    # Plotting centroids
        plt.title('Kmeans clustering')                                                          # Set plot title
        plt.xlabel('X')                                                                         # Set label for x-axis
        plt.ylabel('y')                                                                         # Set label for y-axis
        # Making legends smaller to fit in plot if clusters are >=6
        if k_clusters >=6:                                                                      
            plt.legend(prop={'size': 6})
        else:
            plt.legend()
        fig = plt.gcf()                                                                         # Get the current figure
        fig.set_size_inches((7.5, 5),forward = False)                                           # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                        # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_KmeanClustering.png'),bbox_inches='tight')     # Saving the plot
        plt.clf()                                                                               # Cleaning the current figure after saving

    def write_results(self,dataset,Centroids):
        '''
        ========================================================================
        Description:
        Writting the final results of the centroids and a result tabular with the details of all the features and their distance between each centroids
        and which clusters they belong to. In addition, results of cluster summary is also written.  
        ------------------------------------------------------------------------
        Parameters:
        dataset: Returns a dataframe of extracted features updated with additional columns of distance between the features and the centroids and the
        information of which cluster each feature belongs to; Size -> [No of samples x two independent features + no of centroids + 1(cluster details)]
        Centroids: The mean value of each clusters; Size -> [No of centroids x two independent features]
        ------------------------------------------------------------------------
        Conditions:
        Make sure tabulate package is installed or pip install tabulate
        ------------------------------------------------------------------------
        Output:
        Creating a directory if one does not exists: test_results and writes files: resultKMeans.txt and summaryKMeans.txt      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #-----------------------------------------------------------------------
        if not os.path.exists('test_results'):
                os.makedirs('test_results')
        save_path = os.path.abspath('test_results') # Save the file to the created directory

        # Writting the final results in a tablulated form into the file -> resultKMeans.txt
        #-----------------------------------------------------------------------
        with open(os.path.join(save_path, 'resultKMeans.txt'), 'w') as f:

            print('Resultant centroid coordinates of the cluster:\n',file=f)                                # Creating title infos before writing
            print(f'X_coordinate: {Centroids[0].values},\n'f'y_coordinate: {Centroids[1].values} ',file=f)  # Writing centroid results
            print('\nResultant dataset:\n',file=f)
            print(dataset.to_markdown(tablefmt='grid',index=False),file=f)                                  # Tabulating the results in grid format without index
            print('\nAbbreviations:\n',file=f)                                                              # Adding required abbreviations
            print('PC1,PC2: Extracted features after Principal Component Analysis(PCA)',file=f)
            print('Dist_Centroid[1,2,..n]: Distance between the PC and centroid[1,2,..n]',file=f)
            print('Cluster: The cluster to which the PC coordinates belongs w.r.t distance',file=f)

        # Creating Summary report of the clusters
        #-----------------------------------------------------------------------
        clusters = []                                                                       # Creating storage to store clusters
        observation = []                                                                    # Creating storage to store no of observation for each clusters
        max_dist_Centroid = []                                                              # Creating stoarge to store max distance from centroid
        avg_dist_Centroid = []                                                              # Creating storage to store avg distance from centroid
        no_of_clusters = int(max(dataset['Cluster'].values))                                # Total number of clusters
        for cluster in np.arange(1,no_of_clusters+1,1):
            results = dataset[dataset['Cluster']==cluster]['Dist_Centroid'+str(cluster)]    # Extracting data from final results
            clusters.append('cluster'+str(cluster))                                         # Storing clusters
            observation.append(len(results))                                                # Storing observations
            max_dist_Centroid.append(max(results))                                          # Storing max distance from centroid
            avg_dist_Centroid.append(np.mean(results))                                      # storing avg distance from centroid
        
        # Creating a dataframe for summary so that it can be written in a tablulated form
        #------------------------------------------------------------------------
        summary = pd.DataFrame({'Clusters':clusters,'Observation':observation,'Avg_Dist':avg_dist_Centroid,'Max_Dist':max_dist_Centroid})

        # Writing the Summary results into the file -> summaryKMeans.txt
        #-----------------------------------------------------------------------
        with open(os.path.join(save_path, 'summaryKMeans.txt'), 'w') as f:

            print('K-means Cluster Analysis:\n',file=f)                                        # Creating title infos before writing
            print('Number of clusters: %d'%(no_of_clusters),file=f)                            # Writing total no of clusters
            print('Dataset: Principal Components 1 and 2\n',file=f)                            # Writing dataset details
            print('Observation: Total number of observation for each cluster',file=f)          # Abbreviation for Observation
            print('Avg_Dist: Average distance from centroid',file=f)                           # Abbreviation for Avg_Dist
            print('Max_Dist: Maximum distance from centroid\n',file=f)                         # Abbreviation for Max_Dist
            print(summary.to_markdown(tablefmt='grid',index=False),file=f)                     # Tabulating the results in grid format without index

# Initializing the class 
#-----------------------------------------------------------------------
n_clusters = Elbow_method()  # Elbow method to determine optimal number of clusters