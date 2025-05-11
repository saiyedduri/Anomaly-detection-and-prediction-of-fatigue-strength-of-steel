'''

Creates correlation heatmap w.r.t the input dataset and correlation plot of 
independent features w.r.t its correlation with the target feature
========================================================================
'''
# Import libraries
#------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    Data preprocessing is done to transform raw input data into an understandable and readable format.
    ======================================================================== 
    '''

    def import_dataset(self,data):
        '''
        ========================================================================
        Description:
        Reading in the input dataset with pandas read csv.
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
        dataset: Dataframe of both independent and dependent features; Size -> [No of samples x  features]
        ======================================================================== 
        '''
        dataset = pd.read_csv(data) # Reading in the input dataset
        return dataset

    def feature_scaling(self,dataset):
        '''
        ========================================================================
        Description:
        Standardizing the dataset into a unifrom scale.
        ------------------------------------------------------------------------
        Parameters:
        dataset: Dataframe of both independent and dependent features; Size -> [No of samples x  features]
        ------------------------------------------------------------------------
        Return:
        scaled_dataset: Dataframe of uniformly scaled features; Size -> [No of samples x  features]
        ========================================================================
        '''
        mean_X, std_X = np.array([np.mean(dataset,axis =0)]), np.array([np.std(dataset,axis =0)]) # Mean computed row wise for the dataset(@axis = 0)
        scaled_dataset = (dataset - mean_X)/std_X # Subtracting the original data from its mean value and dividing by its std.
        return scaled_dataset   
#
# ======================================================================
# Construction of correlation matrix
# ======================================================================
#
class Create_correlationMatrix:
    '''
    ========================================================================
    Description:
    Creating correlation matrix to find correlation between the features.
    ======================================================================== 
    '''

    def plot_correlation(self,scaled_dataset,dataset,target_col,y_label,ticks,pltsize):
        '''
        ========================================================================
        Description:
        Creating correlation heatmap and correlation bar chart between features and target feature.
        ------------------------------------------------------------------------
        Parameters:
        scaled_dataset: Dataframe of uniformly scaled features; Size -> [No of samples x  features]
        dataset: Dataframe of both independent and dependent features; Size -> [No of samples x  features]
        target_col: The header of the target column from the dataset; dtype -> str
        y_label: Target feature as input; dtype -> str
        ticks: y_ticks input for plot -> plot_Correlationwith_target.png
        pltsize: The resultant size of the plot
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Plots_CMatrix and returns plot1: plot_AllCorrelation.png and plot2: plot_Correlationwith_target.png 
        ========================================================================
        '''
        self.dataset = dataset
        self.target_col = target_col   

        # Creating correlation on scaled_dataset using pandas correlation
        #------------------------------------------------------------------------
        Correlation_Mat = scaled_dataset.corr()

        # Creating correlation data for correlation between target feature and independent features
        #------------------------------------------------------------------------
        self.data = Correlation_Mat[self.target_col].sort_values(ascending = False)  # Arranging data from higher correlation with target feature to lower correlation 

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_CMatrix'):
            os.makedirs('Plots_CMatrix')
        save_path = os.path.abspath('Plots_CMatrix') # Save the file to the created directory

        # Ploting heat map of all features correlation
        #------------------------------------------------------------------------
        plt.figure(1,figsize = (20,20))                                                     # Set figure size
        cm = sns.heatmap(Correlation_Mat,annot = True,cbar_kws={'label': 'Correlation values'})    # Plot correlation heat map with correlation values
        plt.title('Correlation matrix')                                                     # Set plot title 
        plt.xticks(rotation = 90)                                                           # Rotate xticks by 90°
        plt.yticks(rotation = 0)                                                            # Rotate yticks by 0°
        fig = plt.gcf()                                                                     # Get the current figure
        fig.set_size_inches((pltsize[0], pltsize[1]),forward = False)                       # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                    # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_AllCorrelation.png'),bbox_inches='tight')  # Saving the plot
        plt.clf()                                                                           # Cleaning the current figure after saving

        # Ploting correlation between independent feature and dependent feature
        #------------------------------------------------------------------------
        plt.figure(2)
        plt.bar(list(self.data.keys()[1:]),list(self.data[1:]),label='correlation')         # Plot correlation bar chart of independent features in descending order

        for i, v in enumerate(self.data[(self.data>=0)][1:]):
            plt.text(list(self.data[(self.data>=0)][1:].keys())[i], v+0.05, str('%0.2f '%v),rotation = 90, color='black', fontweight='bold') # Write correlation values for positive correlations

        for i, v in enumerate(self.data[(self.data<0)]): 
            plt.text(list(self.data[(self.data<0)].keys())[i], v-0.17, str('%0.2f '%v),rotation = 90, color='black', fontweight='bold') # Write correlation values for negative correlations
       
        plt.title('Correlation based feature ranking')                                      # Set plot title 
        plt.xticks(rotation = 90)                                                           # Rotate xticks by 90°
        plt.yticks(np.arange(ticks[0],ticks[1],ticks[2]))                                   # Set yticks range as per user input
        plt.xlabel('Independent features')                                                  # Set label for x-axis -> Independent features
        plt.ylabel('Correlation with %s'%(y_label))                                         # Set label for y-axis -> Dependent feature based on user input
        plt.grid(axis = 'y')                                                                # Make grids in y axis
        plt.legend()
        fig = plt.gcf()                                                                            # Get the current figure
        fig.set_size_inches((pltsize[2],pltsize[3]),forward = False)                               # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                           # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path,'plot_Correlationwith_target.png'),bbox_inches='tight') # Saving the plot
        plt.clf()                                                                                  # Cleaning the current figure after saving

    def write_selectfeatures(self,limit):
        '''
        ========================================================================
        Description:
        To create a new dataset by filtering out features that have negative correlation with the dependent feature.
        This new dataset can be used in other ML models for prediction.
        ------------------------------------------------------------------------
        Parameters:
        limit: The range of features to write; dtype -> int
        ------------------------------------------------------------------------
        Conditions:
        Only positive valued features can be written
        Features can be selected only in the given order
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Results_CMatrix and writes a dataset: fatigue_Selecteddataset.csv      
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_CMatrix'):
            os.makedirs('Results_CMatrix')
        save_path = os.path.abspath('Results_CMatrix') # Save the file to the created directory

        selected = self.data[(self.data>=0)][0:limit+1].keys() # Selecting positive valued features
        selected = list(selected)[1:] + [list(selected)[0]]    # Moving dependent feature to the last column
        selected_dataset = self.dataset[selected]              # Creating dataset based on selection
        selected_dataset.to_csv(os.path.join(save_path, 'fatigue_Selecteddataset.csv'),index=False) # Writing selected features into a .csv file