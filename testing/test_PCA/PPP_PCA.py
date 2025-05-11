'''
========================================================================
Library_name: PPP_PCA
========================================================================
'''
# Import required libraries
#------------------------------------------------------------------------
import pandas as pd
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
import os
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
 
    def PCA(self,scaled_X):
        '''
        ========================================================================
        Description:
        Extracting features using Principal Component Analysis(PCA).
        ------------------------------------------------------------------------
        Parameters:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        ------------------------------------------------------------------------
        Return:
        extracted_X: Extracted N independent features into selected no of features; Size -> [No of samples x selected no of PC] 
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

        selected_PC = 2                                                 # For testing selected PC will be assigned as 2
        self.extracted_eigen_vec = self.sort_eigen_vec[:,0:selected_PC] # Extracting the features w.r.t selected no of PCs; Size -> [independent features x selected no of PC]
       
        # Projecting n-dim dataset into 2-dim feature space
        #------------------------------------------------------------------------
        # PCA is a linear combination of loading vector(extracted_eigen_vec) and the observation vector(scaled_X)
        # Computing the Score which represents the projection of i_th observation onto the j_th PC
        # Each column in the extracted_X represents the score

        extracted_X = np.dot(scaled_X,self.extracted_eigen_vec)  # Extracted features by transforming the data into new subspace; Size -> [No of samples x selected no of PC]
        return extracted_X,mat_Covariance,self.sort_eigen_val,self.extracted_eigen_vec
        
    def PCA_results(self,plt_size):
        '''
        ========================================================================
        Description:
        Writing testing results to validate with the expected results. 
        ------------------------------------------------------------------------
        Parameters:
        plt_size: Plot size for Scree plot
        ------------------------------------------------------------------------
        Return:
        Creates a directory: test_results and write file: testResults.txt and plot: plot_Scree.png 
        ========================================================================
        ''' 
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('test_results'):
                os.makedirs('test_results')
        save_path = os.path.abspath('test_results')     # Save the file to the created directory

        # Creating a dataframe to write the Loading vector and varience for first 2 max varience in a tablulated form into the file -> testResults.txt
        #------------------------------------------------------------------------
        dataframe = {}                                                                      # Creating a dataframe
        dataframe['Features'] = self.scaled_X.keys()                                        # Writing features into the dataframe
        for pc in np.arange(0,len(self.extracted_eigen_vec[:2]),1):
           dataframe['Loading_vector'+str(pc+1)] = self.extracted_eigen_vec[:,pc]           # Writing entries into the dataframe
        dataframe['Variance(%)'] = (self.sort_eigen_val/np.sum(self.sort_eigen_val))*100    # Writing variance results in percentage
        writedata = pd.DataFrame(dataframe)                                         

        # Writing the Loading results into the file -> testResults.txt
        #-----------------------------------------------------------------------
        with open(os.path.join(save_path, 'testResults.txt'), 'w') as f:

            print('Test results of PCA for all features:\n',file=f)                         # Creating title infos before writing             
            print('Loading_vector: Eigen vectors of the covariance matrix; Max varience i.e. 1 & 2 of these vectors are listed below',file=f)      
            print('Variance: It is the variance of all %d Eigen values in percentage'%(len(self.sort_eigen_val)),file=f)
            print('\nThe details below are used for Biplot analysis:\n',file=f)   
            print( writedata.to_markdown(tablefmt='grid',index=False),file=f)               # Tabulating the results in grid format without index

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
        Creates a directory: test_results and returns a plot: plot_PCA.png      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        if not os.path.exists('test_results'):
                os.makedirs('test_results')
        save_path = os.path.abspath('test_results') # Save the file to the created directory

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
            # To get the exact results of the toy dataset, sorted Eigen values are standardize
            std_sort_eigen_val = np.sqrt(self.sort_eigen_val)
            plt.arrow(0,0,self.extracted_eigen_vec[:,0][i]*(std_sort_eigen_val[0]/0.5),self.extracted_eigen_vec[:,1][i]*(std_sort_eigen_val[1]/0.5),head_width=0.2,color=c,label=txt)
            plt.legend(loc= 'upper left',prop={'size': 6},bbox_to_anchor=(1,1))
        fig = plt.gcf()                                                                                 # Get the current figure
        fig.set_size_inches((plt_size[2], plt_size[3]),forward = False)                                 # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                                # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_PCA.png'),bbox_inches='tight')                         # Saving the plot
        plt.clf()                                                                                       # Cleaning the current figure after saving