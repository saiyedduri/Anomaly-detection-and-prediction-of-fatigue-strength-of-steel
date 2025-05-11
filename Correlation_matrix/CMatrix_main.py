'''
========================================================================
This program is a part of personal programming project
file: PPP_CMatrix_main
------------------------------------------------------------------------
Creates correlation heatmap with input dataset
Creates a correlation plot for independent features w.r.t dependent features 
Creates a dataset w.r.t selected features which has positive correlations with dependent features
condition: While writing fatigue_Selecteddataset.csv, only positive valued features can be writen and 
features can only be selected in the given order as in the plot -> plot_Correlationwith_target.png
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_CMatrix import Data_preprocessing
from PPP_CMatrix import Create_correlationMatrix

# Import required libraries
# -------------------------
import click
#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
@click.command()
@click.option('--data',nargs=1,type=str,default='fatigue_dataset.csv',help='Enter input dataset.csv: last column must be the target feature')
@click.option('--target_column',nargs=1,type=str,default='Fatigue',help='Enter Target column header from dataset')
@click.option('--targetf',nargs=1,type=str,default='Fatigue strength',help='Enter Target feature')
@click.option('--yticks',nargs =3,type=float,default=([-1,1.3,0.2]),help='Enter yticks for plot_Correlationwith_target')
@click.option('--selected_features',nargs=1,type=int,default=12,help='Select from plot_Correlationwith_target the number of features to write')
@click.option('--plt_size',nargs=4,type=float,default=([22,9,9.5, 7]),help='Enter plot size for AllCorr. & Corrwith_targ.')

def correlationMatrix(data,target_column,targetf,yticks,selected_features,plt_size):
    '''
    ========================================================================
    Description:
    This correlation plots gives information regarding the features that influences the target feature. Now we can select and remove features based
    on their positive and negative influence on the target feature. Note that w.r.t the dataset the target_column and targetf should be modified. Based
    on the number of features the size of the plot can also be modified. 
    ========================================================================
    '''
    # Initialization for Correlation matrix
    #------------------------------------------------------------------------
    Preprocessed_data = Data_preprocessing()                    # Preprocessing the dataset
    CMatrix = Create_correlationMatrix()                        # Creating correlation matrix

    # Data_preprocessing
    #------------------------------------------------------------------------
    dataset = Preprocessed_data.import_dataset(data)            # Reading in the dataset as csv
    scaled_dataset = Preprocessed_data.feature_scaling(dataset) # Scaling the dataset

    # Creating correlation heatmap and plot between the features and target feature
    #------------------------------------------------------------------------
    CMatrix.plot_correlation(scaled_dataset,dataset,target_column,targetf,yticks,plt_size) # Creating correlation plots

    # Creating dataset based on correlation
    #------------------------------------------------------------------------
    CMatrix.write_selectfeatures(selected_features)             # Creating dataset -> fatigue_Selecteddataset.csv

if __name__ == '__main__':
    correlationMatrix()