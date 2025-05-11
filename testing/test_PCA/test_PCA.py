'''
========================================================================
Test_file: test_PCA.py
Test_type: Testing with sample dataset
Aim: To check the correctness of implementation the model is tested with known dataset and verified whether it is able to recreate the expected results
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_PCA import Data_preprocessing
from PPP_PCA import Feature_extraction

# Import required libraries
# -------------------------
import click
import numpy as np

#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
def test_PCA():
    @click.command()
    @click.option('--data',nargs=1,type=str,default='sample-data_pca.csv',help='Enter input dataset.csv: last column must be the target feature')
    @click.option('--plt_size',nargs=4,type=float,default=([7.5, 5,7.5, 5]),help='Enter plot size for plot_Scree & plot_PCA')
    #
    # ============================================================================================================================================
    #                                                      Validating --> Principal Component Analysis
    # ============================================================================================================================================
    #
    def PCA(data,plt_size):
        '''
        ========================================================================
        Description:
        Test for principal component analysis; Check obtained results 'plot_PCA.png', 'plot_Scree.png', and 'testResults.txt' from the 
        directory test_results with the expected results from the directory expected_results.
        ========================================================================
        '''
        # Initialization for PCA
        #------------------------------------------------------------------------
        Preprocessed_data = Data_preprocessing()                            # Feature scaling
        extracting_X = Feature_extraction()                                 # Feature extraction

        # Data_preprocessing
        #------------------------------------------------------------------------                                        
        X = Preprocessed_data.import_dataset(data)                          # X: Independent feature
        scaled_X = Preprocessed_data.feature_scaling(X)                     # Scaled features

        # Principal Component Analysis(PCA) 
        #------------------------------------------------------------------------
        extracted_X,mat_Covariance,eigen_val,extracted_eigen_vec = extracting_X.PCA(scaled_X) # Creating extracted features

        # Principal Component Analysis(PCA) method results 
        #------------------------------------------------------------------------
        extracting_X.PCA_results(plt_size)                                  # Writing test results, Creating scree plot

        # Plot to analyse extracted features
        #------------------------------------------------------------------------
        extracting_X.plot_PCA(extracted_X,X,plt_size)                       # Ploting Biplot for analysis

        # Checking whether the sum of calculated eigenvalues are equal to the total variance of the input data
        #------------------------------------------------------------------------
        sum_eigen_val = np.sum(eigen_val)                                   # Sum of the eigen values
        total_variance = np.sum(np.diagonal(mat_Covariance))                # Total varience of the input data
        np.testing.assert_almost_equal (total_variance, sum_eigen_val)
        print('\nAssertionPassed: Sum of calculated eigenvalues %f == total variance of the data %f'%(sum_eigen_val,total_variance))

        # Checking whether the length of the computed eigen vector is 1
        #------------------------------------------------------------------------
        np.testing.assert_almost_equal (np.sum(extracted_eigen_vec[:,1]**2), 1)   # Test for principal components
        print('\nAssertionPassed: The length of the computed eigen vector is = 1\n')

    if __name__ == '__main__':
        PCA()

test_PCA()