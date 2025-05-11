'''
========================================================================
Machine learning algorithm: K-Means Clustering(unsupervised machine learning algorithm for Clustering)
To implemented k-Means algorithm and find there exist any pattern within the features(here parameters from different heats)
To implement feature extraction algorithm PCA to reduce the no of features to 2 for better visualization and to perform clustering 
Elbow method to determine the optimal number of clusters
========================================================================
'''
# Import required libraries
#------------------------------------------------------------------------
import pandas as pd
import numpy as np
import scipy.linalg as spl
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
        Reading in the input dataset and separating it into X:independent features and y:dependent feature.
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
        y: Dataframe of dependent feature; Size -> [No of samples]
        ======================================================================== 
        '''
        dataset = pd.read_csv(data)  # Reading in the input dataset
        X = dataset.iloc[:,:-1]      # Selecting all the columns of the input dataset except the last(which is the dependent feature/ target feature)
        y = dataset.iloc[:,-1]       # Selecting the last column of the input dataset
        return X,y

    def feature_scaling(self,X,y):
        '''
        ========================================================================
        Description:
        Standardizing the independent and dependent features into a unifrom scale.
        ------------------------------------------------------------------------
        Parameters:
        X: Dataframe of independent features; Size -> [No of samples x independent features]
        y: Dataframe of dependent feature; Size -> [No of samples]
        ------------------------------------------------------------------------
        Check:
        The mean of the standardized data = 0 (close enough)
        The standard deviation of the standardized data = 1 (close enough)
        ------------------------------------------------------------------------
        Return:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        scaled_y: Dataframe of uniformly scaled dependent feature; Size -> [No of samples]
        ========================================================================
        '''
        mean_X, std_X = np.array([np.mean(X,axis =0)]), np.array([np.std(X,axis =0)]) # Mean computed row wise for independent features(@axis = 0)
        scaled_X = (X - mean_X)/std_X                                                 # Subtracting the original data from its mean value and dividing by its std.
        mean_y, std_y = np.array([np.mean(y)]), np.array([np.std(y)])
        scaled_y = (y - mean_y)/std_y
        return scaled_X, scaled_y
#
# ======================================================================
# Construction of Feature extraction
# ======================================================================
#
class Feature_extraction:
    '''
    ========================================================================
    Description:
    Feature Extraction -> reducing the dimensionality of the features for visualization and analysis.
    ======================================================================== 
    '''
 
    def PCA(self,scaled_X,selected_PC):
        '''
        ========================================================================
        Description:
        Extracting features using Principal Component Analysis(PCA).
        ------------------------------------------------------------------------
        Parameters:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        selected_PC: Selected number of principal components
        ------------------------------------------------------------------------
        Return:
        extracted_X: Extracted N independent features into selected no of features; Size -> [No of samples x selected no of PC]
        self.sort_eigen_val: Eigen value sorted from high variance to low   
        ------------------------------------------------------------------------
        Note:
        -rowvar = False -> each column represents a variable, while the rows contain observations.
        ========================================================================
        '''
        self.scaled_X = scaled_X     
        mat_Covariance = np.cov(scaled_X, rowvar = False)    # Calculating the covariance of the scaled data; Size -> [independent features x independent features]
        eigen_val,eigen_vec = spl.eigh(mat_Covariance)       # Calculating the eigen values and the eigen vectors of the covariance matrix
        # eigen vector; size -> [independent features x independent features]
        # eigen value; size -> [independent features]
        # The eigen vectors of mat_Covariance are orthogonal to each other and each vector represents the principal axis(here it is a column vector) 
        # The eigen value represents the variance of those vectors
                           
        # Sorting eigen values and eigen vectors from high variance to low
        #------------------------------------------------------------------------
        sort_index = np.argsort(eigen_val)[::-1]                        # Sorting in descending to get high variance to low and returning their index
        self.sort_eigen_val = eigen_val[sort_index]                     # Eigen value sorted from high variance to low
        self.sort_eigen_vec = eigen_vec[:,sort_index]                   # Eigen vector(load vector) sorted from max eigen vector and its magnitude will the max variance

        # Select number of principal components
        #------------------------------------------------------------------------
        # Method 1
        varience_explained = np.cumsum(self.sort_eigen_val/np.sum(self.sort_eigen_val))                  # Computing the varience explained criteria
        self.method1_res = [idx for idx, varience in enumerate(varience_explained) if varience >= 0.80]  # Selecting the no of PC w.r.t 80% threshold
        # Method 2
        self.method2_res = self.sort_eigen_val[self.sort_eigen_val>=1]                                   # Selecting no of PC w.r.t kaiser rule
        # User selection
        self.extracted_eigen_vec = self.sort_eigen_vec[:,0:selected_PC] # Extracting the features w.r.t selected no of PCs; Size -> [independent features x selected no of PC]

        # Projecting n-dim dataset into 2-dim feature space
        #------------------------------------------------------------------------
        # PCA is a linear combination of loading vector(extracted_eigen_vec) and the observation vector(scaled_X)
        # Computing the Score which represents the projection of i_th observation onto the j_th PC
        # Each column in the extracted_X represents the score
        extracted_X = np.dot(scaled_X,self.extracted_eigen_vec)  # Extracted features by transforming the data into new subspace; Size -> [No of samples x selected no of PC]
        return extracted_X,self.sort_eigen_val
        
    def PCA_results(self,plt_size):
        '''
        ========================================================================
        Description:
        Writing results of methods to select principal components and Loading result(eigen value and eigen vector) at PC1 and PC2. 
        In addition Scree plot is also created to select principal components.
        ------------------------------------------------------------------------
        Parameters:
        plt_size: Plot size for Scree plot
        ------------------------------------------------------------------------
        Return:
        Creates a directory: Results_KMeans and writes files: SelectPrincipalComponents.txt and LoadingResults.txt 
        Creates a directory: Plots_KMeans and returns a plot: plot_Scree.png     
        ========================================================================
        ''' 
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_KMeans'):
                os.makedirs('Results_KMeans')
        save_path = os.path.abspath('Results_KMeans')     # Save the file to the created directory

        # Selecting method to choose the number of Principal Components for extraction from the file SelectPrincipalComponents.txt 
        # writing results of two methods into the file -> SelectPrincipalComponents.txt
        #------------------------------------------------------------------------
        with open(os.path.join(save_path,'SelectPrincipalComponents.txt') , 'w') as f:
        
            print('Select any one method from below to choose number of principal components to extract:',file=f)       # Creating title infos before writing
            print('\nMethod1: The varience explained criteria\n', file=f)                                               # Title of method 1
            print(np.cumsum(self.sort_eigen_val/np.sum(self.sort_eigen_val)),file=f)                                    # The varience explained criteria
            print('\nRecommended number of principal components to select = %d'%(self.method1_res[0]+1), file=f)        # Result for method 1
            print('\nNote: The number of principal components are selected w.r.t a threshold of 80%(0.80)', file=f)     # Note for method 1
            print('\n#------------------------------------------------------------------------', file=f) 
            print('\nMethod2: Kaiser-Guttman criterion(frequently used)\n', file=f)                                     # Title of method 2
            print(self.sort_eigen_val[self.sort_eigen_val>=1], file=f)                                                  # The kaiser rule
            print('\nRecommended number of principal components to select = %d'%(len(self.method2_res)), file=f)        # Result for method 2
            print('\nNote: Principal components with varience > 1 are selected', file=f)                                # Note for method 2
            print('\n#------------------------------------------------------------------------', file=f) 
            print('\nMethod3: Scree plot\n', file=f)                                                                    # Method 3 Scree plot
            print('Note: Select from plot_Scree.png present in the directory Plots_KMeans', file=f)                     # Result for method 2

        # Creating a dataframe to write the Loading vector and values for first 2 max varience in a tablulated form into the file -> LoadingResults.txt
        #------------------------------------------------------------------------
        dataframe = {}                                                                      # Creating a dataframe
        dataframe['Features'] = self.scaled_X.keys()                                        # Writing features into the dataframe
        for pc in np.arange(0,len(self.extracted_eigen_vec[:2]),1):
           dataframe['Loading_vector'+str(pc+1)] = self.extracted_eigen_vec[:,pc]           # Writing entries into the dataframe
        dataframe['Loading_value'] = self.sort_eigen_val                                    # Writing loding(eigen) values into the dataframe
        dataframe['Variance(%)'] = (self.sort_eigen_val/np.sum(self.sort_eigen_val))*100    # Writing variance results in percentage
        writedata = pd.DataFrame(dataframe)                                         

        # Writing the Loading results into the file -> LoadingResults.txt
        #-----------------------------------------------------------------------
        with open(os.path.join(save_path, 'LoadingResults.txt'), 'w') as f:

            print('Loading results of PCA for all features:\n',file=f)                   # Creating title infos before writing           
            print('The abbreviation of Features shall be referred from report',file=f)   
            print('Loading_vector: Eigen vectors of the covariance matrix; Max varience i.e. 1 & 2 of these vectors are listed below',file=f)      
            print('Loading_value: All %d Eigen values of the covariance matrix are listed below'%(len(self.sort_eigen_val)),file=f)
            print('Variance: It is the variance of all %d Eigen values in percentage'%(len(self.sort_eigen_val)),file=f)
            print('Vectors of each features in the Biplot is plotted by(x/y) -> Loading_vector[1/2] * Loading_value[0/1]/0.5\n',file=f) 
            print('The details below are used for Biplot analysis:\n',file=f)   
            print( writedata.to_markdown(tablefmt='grid',index=False),file=f)            # Tabulating the results in grid format without index

        # Creating a directory if one does not exists
        if not os.path.exists('Plots_KMeans'):
                os.makedirs('Plots_KMeans')
        save_path = os.path.abspath('Plots_KMeans') # Save the file to the created directory

        # Creating Scree plot, also one of the method to select PC but this is not reliable, thus only for reference purpose 
        #------------------------------------------------------------------------
        self.varience_eigen_val = self.sort_eigen_val/np.sum(self.sort_eigen_val)       # Calculating the proportion of varience of the Eigen values
        plt.title('Scree plot')                                                         # Set plot title
        plt.xlabel('Principal components')                                              # Set label for x-axis -> Principal components
        plt.ylabel('Proportion of varience')                                            # Set label for y-axis -> Proportion of varience
        plt.plot(np.arange(1,len(self.sort_eigen_val)+1,1),abs(self.varience_eigen_val),'o-',label='varience')  # Scree plot(Elbow) of proportion of varience
        plt.xticks(np.arange(1,len(self.sort_eigen_val)+1,1))                           # Assinging X-ticks for PC
        plt.legend()
        fig = plt.gcf()                                                                 # Get the current figure
        fig.set_size_inches((plt_size[0], plt_size[1]),forward = False)                 # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_Scree.png'),bbox_inches='tight')       # Saving the plot
        plt.clf()                                                                       # Cleaning the current figure after saving

    def plot_PCA(self,extracted_X,X,plt_size):
        '''
        ========================================================================
        Description:
        PCA Analysis with Correlation Biplot with Principal Component1(PC1) and Principal Component2(PC2). 
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into selected no of features; Size -> [No of samples x selected no of PC]
        X: Dataframe of independent features; Size -> [No of samples x independent features]
        plt_size: Plot size for PCA plot
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Plots_KMeans and returns a plot: plot_PCA.png      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        if not os.path.exists('Plots_KMeans'):
                os.makedirs('Plots_KMeans')
        save_path = os.path.abspath('Plots_KMeans') # Save the file to the created directory

        # PCA Analysis with Correlation Biplot -> PC scores + loading vectors
        #------------------------------------------------------------------------
        plt.title('PCA analysis with Correlation Biplot')                                               # Set plot title
        plt.xlabel('PC1({0:.1f}%)'.format(self.varience_eigen_val[0]*100))                              # Set label for x-axis -> will be the Principal Component 1
        plt.ylabel('PC2({0:.1f}%)'.format(self.varience_eigen_val[1]*100))                              # Set label for y-axis -> will be the Principal Component 2
        plt.scatter(extracted_X[:,0],extracted_X[:,1],marker='o',facecolors='none',edgecolors='black',label='Scores')  # Plotting biplot
        plt.axvline(x=0,color='green')                                                                  # PC 1 axis  
        plt.axhline(y=0,color='blue')                                                                   # PC 2 axis
        colors=iter(plt.cm.nipy_spectral(np.linspace(0,1,len(self.extracted_eigen_vec))))               # Colormap for each features                                                         
        for i,txt in enumerate(X.keys()):              
            c = next(colors)                                                                            # Selecting next color for each iteration
            # Plotting features as vectors and considering Factor 0.5 to make the arrow more obvious
            plt.arrow(0,0,self.extracted_eigen_vec[:,0][i]*(self.sort_eigen_val[0]/0.5),self.extracted_eigen_vec[:,1][i]*(self.sort_eigen_val[1]/0.5),head_width=0.2,color=c,label=txt)
            plt.legend(loc= 'upper left',prop={'size': 6},bbox_to_anchor=(1,1.15))
        fig = plt.gcf()                                                                                 # Get the current figure
        fig.set_size_inches((plt_size[2], plt_size[3]),forward = False)                                 # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                                # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_PCA.png'),bbox_inches='tight')                         # Saving the plot
        plt.clf()                                                                                       # Cleaning the current figure after saving

    def write_extractedFeatures(self,extracted_X,scaled_y,target):
        '''
        ========================================================================
        Description:
        Creating a dataset using the extrated features with the help of pandas dataframe.
        ------------------------------------------------------------------------
        Parameters:
        extracted_X: Extracted N independent features into selected no of features; Size -> [No of samples x selected no of PC]
        scaled_y: Dataframe of uniformly scaled dependent feature; Size -> [No of samples]
        target: Target column header from extracted dataset
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Results_KMeans and returns a dataset: fatigue_Extracteddataset.csv      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_KMeans'):
                os.makedirs('Results_KMeans')
        save_path = os.path.abspath('Results_KMeans')     # Save the file to the created directory

        # Creating a dataframe to write the extracted features into .csv
        #------------------------------------------------------------------------
        dataframe = {}                                                              # Creating a dataframe
        for pc in np.arange(0,len(extracted_X[0]),1):
           dataframe['PC'+str(pc+1)] = extracted_X[:,pc]                            # Writing entries into the dataframe
        dataframe[target] = scaled_y                                                # Writing target feature into the dataframe
        writedata = pd.DataFrame(dataframe)                                         
        writedata.to_csv(os.path.join(save_path, str(target)+'_Extracteddataset.csv'),index = False)  # Creating a extracted dataset w.r.t selected no of PC
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
        Creating a directory if one does not exists: Plots_KMeans and returns a plot: plot_ElbowMethod.png      
        ========================================================================
        '''
        K = np.arange(1,k+1,1)          # Plotting for different number of centroids                                      
        WCSS = []                       # Creating storage for within-clusters sum of squares
        for value in K:
            sum_distance = n_clusters.determine_clusters(extracted_X,value) # Calculating within-clusters sum of squares
            WCSS.append(sum_distance)   # Storing the value of within-clusters sum of squares for each number of centroids

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_KMeans'):
                os.makedirs('Plots_KMeans')
        save_path = os.path.abspath('Plots_KMeans') # Save the file to the created directory
        
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
        Extracted dataset with two Principal Components are used to make clusters.
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

    def plot_Kmean_clustering(self,k_clusters,extracted_X,Centroids,scaled_X,sort_eigen_val):   
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
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        sort_eigen_val: Eigen values of the covarience matrix; Size -> [independent features]
        ------------------------------------------------------------------------
        Output:
        Creating a directory if one does not exists: Plots_KMeans and returns two plots: plot_KmeanClustering.png and plot_parallelCoordinates.png      
        ========================================================================
        '''   
        # Creating a directory if one does not exists
        #-----------------------------------------------------------------------
        if not os.path.exists('Plots_KMeans'):
                os.makedirs('Plots_KMeans')
        save_path = os.path.abspath('Plots_KMeans') # Save the file to the created directory

        for k in range(k_clusters):
            dataset=extracted_X[extracted_X['Cluster']==k+1]      # To plot each cluster in different colour, each cluster will be ploted one after another
            plt.scatter(dataset['PC1'],dataset['PC2'],label='cluster %d'%(k+1))
        
        # Plot K-Means clustering     
        #------------------------------------------------------------------------
        varience_eigen_val = sort_eigen_val/np.sum(sort_eigen_val)                              # Calculating the proportion of varience of the Eigen values
        plt.scatter(Centroids[0],Centroids[1], c='black',marker='*',s=150,label='centroids')    # Plotting centroids
        plt.title('Kmeans clustering')                                                          # Set plot title
        plt.xlabel('PC1({0:.1f}%)'.format(varience_eigen_val[0]*100))                           # Set label for x-axis -> will be the Principal Component 1
        plt.ylabel('PC2({0:.1f}%)'.format(varience_eigen_val[1]*100))                           # Set label for y-axis -> will be the Principal Component 2
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

        # Plot parallel coordinates 
        #------------------------------------------------------------------------                                                                          
        scaled_X['Cluster'] = extracted_X['Cluster']                                            # Storing cluster details into the scaled dataset
        plt.title('Parallel Coordinates of the clusters')                                       # Set plot title
        plt.xticks(rotation = 90)                                                               # Rotating Xticks by 90° for better visualization
        pd.plotting.parallel_coordinates(scaled_X,'Cluster',colormap='viridis')                 # Plotting parallel coordinates
        fig = plt.gcf()                                                                         # Get the current figure
        fig.set_size_inches((9, 5),forward = False)                                             # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                        # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_parallelCoordinates.png'),bbox_inches='tight') # Saving the plot
        plt.clf()                                                                               # Cleaning the current figure after saving 

    def write_results(self,dataset,Centroids):
        '''
        ========================================================================
        Description:
        Writing the final results of the centroids and a result tabular with the details of all the features and their distance between each centroids
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
        Creating a directory if one does not exists: Results_KMeans and writes files: resultKMeans.txt and summaryKMeans.txt      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #-----------------------------------------------------------------------
        if not os.path.exists('Results_KMeans'):
                os.makedirs('Results_KMeans')
        save_path = os.path.abspath('Results_KMeans') # Save the file to the created directory

        # Writing the final results in a tablulated form into the file -> resultKMeans.txt
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





