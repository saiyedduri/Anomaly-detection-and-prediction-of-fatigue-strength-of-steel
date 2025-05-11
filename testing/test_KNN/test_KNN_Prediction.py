'''
========================================================================
Test_type: Unit test
Aim: To validate the prediction of KNN for a given set of input parameters
========================================================================   
'''
# Import required libraries
# -------------------------
import numpy as np
import pytest 
import sys
import os
#
# ======================================================================
# Construction of Dataset
# ======================================================================
# 
class Data_set:
    '''
    ========================================================================
    Description: 
    Creating required datasets.
    ======================================================================== 
    '''
    def data(self,startx,endx,starty,endy):
        '''
        ========================================================================
        Description:
        Creating a range of independent and dependent features based on user inputs
        ------------------------------------------------------------------------
        Parameters:
        startx: Start of the interval for independent feature range; dtype -> int
        endx: End of the interval for independent feature range; dtype -> int
        starty: Start of the interval for dependent feature range; dtype -> int
        endy: End of the interval for dependent feature range; dtype -> int
        ------------------------------------------------------------------------
        Return:
        X: Independentfeature; Array size -> [No of entries x 1]
        y: Dependent feature; Array size -> [No of entries x 1]
        ========================================================================
        '''
        X = np.array([np.arange(startx,endx,1)]).T
        y = np.array([np.arange(starty,endy,1)]).T
       
        return X,y

    def split_dataset(self,X,y):
        '''
        ========================================================================
        Description:
        Splitting the dataset into training and test set for independent and dependent features
        ------------------------------------------------------------------------
        Parameters:
        X: Independentfeature; Array size -> [No of entries x 1]
        y: Dependent feature; Array size -> [No of entries x 1]
        ------------------------------------------------------------------------
        Return:
        X_train: Training set of independent features with 80% of the samples; Array size -> [80% of entries x 1]
        X_test: Testing set of independent features with 20% of the samples; Array size -> [20% of entries x 1]
        y_train: Training set of dependent feature with 80% of the samples; Array size -> [80% of entries x 1]
        y_test: Testing set of dependent feature with 20% of the samples; Array size -> [20% of entries x 1]
        ========================================================================
        '''
        X_train, X_test = np.split(X,[int(0.8*len(X))])
        y_train, y_test = np.split(y,[int(0.8*len(y))])
        return np.array(X_train), np.array(X_test),np.array(y_train),np.array(y_test)
#
# ======================================================================
# Construction of KNN algorithm
# ======================================================================
#
class Knearest_neighbor:
    '''
    ========================================================================
    Description: 
    Implementing KNN algorithm to make prediction of the test set using training set.
    ======================================================================== 
    '''
    def predict_neighbor(self,X_train, X_test,y_train,y_test,k):
        '''
        ========================================================================
        Description:
        To find out dependent feature of each sample from its independent features with the help of a reference dataset.
        ------------------------------------------------------------------------
        Parameters:
        X_train: Training set of independent features with 80% of the samples; Array size -> [80% of entries x 1]
        X_test: Testing set of independent features with 20% of the samples; Array size -> [20% of entries x 1]
        y_train: Training set of dependent feature with 80% of the samples; Array size -> [80% of entries x 1]
        y_test: Testing set of dependent feature with 20% of the samples; Array size -> [20% of entries x 1]
        k: No of samples to be considered as nearest neighbours; dtype -> int
        ------------------------------------------------------------------------
        Conditions:
        No of samples to be considered as nearest neighbours should be greater than zero
        ------------------------------------------------------------------------
        Return:
        y_pred: Predicted dependent feature from x_test independent features; Array size -> [20% of entries x 1]     
        ========================================================================
        '''
        y_pred = np.zeros(y_test.shape)                                           # To store prediction results
        for row in range (len(X_test)):
            eucledian_distance = np.linalg.norm(X_train - X_test[row], axis = 1)  # Finding the norm of the independent features between the reference and prediction dataset
            neighbor_index = eucledian_distance.argsort()[:k]   # Sorting the calculated distance list and selecting the indexes of the smallest norms based on k input
            nearest_neighbor = y_train[neighbor_index]          # Using the selected indexes we find the equivalent dependent feature from the reference dataset
            y_pred[row] = nearest_neighbor.mean()               # Now we calculate the prediction result of each sample by taking the mean of found dependent feature 
        
        return y_pred

def read_parameters():
    '''
    ========================================================================
    Description:
    Reading in parameters such as independent feature range, dependent feature range, prediction results and nearest neighbors for 
    five test cases from the file: parameters.txt in the same directory 
    ------------------------------------------------------------------------
    Return:
    param: Input set for pytest.mark.parametrize to check and varify all the test cases
    ========================================================================
    '''
    indFeat = []                                                    # Creating storage to store independent feature range
    depFeat = []                                                    # Creating storage to store dependent feature range
    expPredRes = []                                                 # Creating storage to store expected prediction results
    n_samples = []                                                  # Creating storage to store selected nearest neighbors

    # Setting path to open parameters file
    #------------------------------------------------------------------------
    cwd = os.path.basename(os.getcwd())                             # Locating current working directory
    if cwd == 'test_KNN':                                           # If tested from the current directory
        filename = 'parameters.txt'
    elif cwd == 'test_ML':                                          # If tested from test_ML
        open_path = os.path.abspath('test_KNN')           
        filename = os.path.join(open_path,'parameters.txt')
    else:                                                           # Else an error will be thrown stating that the user is in the wrong directory
        sys.exit('Error:Testing executed in the wrong directory\nPossible directories: test_ML, test_ANN and current directory of the file')
        
    with open(filename, 'r') as f:                                          # Read parameters
        read_param = f.readlines()
        for idx in np.arange(8,13,1):                                       # Iterating over the fixed no of lines
            read_indFeat = read_param[idx].split(' ')[0]                    # Reading in the fixed no of independent feature range
            read_depFeat = read_param[idx].split(' ')[2]                    # Reading in the fixed no of dependent feature range
            read_expPredRes = read_param[idx].split(' ')[5]                 # Reading in the fixed no of expected result
            read_samples = read_param[idx].split(' ')[11]                   # Reading in the fixed nearest neighbors
            indFeat.append(list(map(int,read_indFeat.split(','))))          # Storing the independent feature range
            depFeat.append(list(map(int,read_depFeat.split(','))))          # Storing the dependent feature range
            expPredRes.append(list(map(float,read_expPredRes.split(','))))  # Storing expected result
            n_samples.append(int(read_samples))                             # Storing nearest neighbors

    # Creating test cases for parametrize
    #------------------------------------------------------------------------
    param = [(indFeat[0][0],indFeat[0][1],depFeat[0][0],depFeat[0][1],np.array([expPredRes[0]]).T,n_samples[0]),(indFeat[1][0],indFeat[1][1],depFeat[1][0],depFeat[1][1],\
    np.array([expPredRes[1]]).T,n_samples[1]),(indFeat[2][0],indFeat[2][1],depFeat[2][0],depFeat[2][1],np.array([expPredRes[2]]).T,n_samples[2]),(indFeat[3][0],indFeat[3][1],
    depFeat[3][0],depFeat[3][1],np.array([expPredRes[3]]).T,n_samples[3]),(indFeat[4][0],indFeat[4][1],depFeat[4][0],depFeat[4][1],np.array([expPredRes[4]]).T,n_samples[4])]
    return param

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('startx,endx,starty,endy,expected,k', read_parameters())

def test_Complete_KNN(startx,endx,starty,endy,expected,k):
    '''
    ========================================================================
    Description:
    Test KNN operation for the given set of input parameters
    ------------------------------------------------------------------------
    Parameters:
    startx: Start of the interval for independent feature range; dtype -> int
    endx: End of the interval for independent feature range; dtype -> int
    starty: Start of the interval for dependent feature range; dtype -> int
    endy: End of the interval for dependent feature range; dtype -> int
    expected: Expected predicted results; Array size -> [1 x No of entries] 
    k: No of samples to be considered as nearest neighbours; dtype -> int
    ========================================================================
    '''
    # Preprocessing the dataset
    #------------------------------------------------------------------------
    Preprocessed_data = Data_set()

    # K_Nearest Neighbor for regression
    #------------------------------------------------------------------------
    Kneighbor = Knearest_neighbor()

    # Data_preprocessing
    #------------------------------------------------------------------------
    X,y = Preprocessed_data.data(startx,endx,starty,endy)                     
    X_train, X_test,y_train,y_test = Preprocessed_data.split_dataset(X,y)

    # Knearest_neighbor
    #------------------------------------------------------------------------
    y_pred= Kneighbor.predict_neighbor(X_train, X_test,y_train,y_test,k) 
    
    # Assertion
    #------------------------------------------------------------------------
    np.testing.assert_array_almost_equal(y_pred,expected), 'test failed'