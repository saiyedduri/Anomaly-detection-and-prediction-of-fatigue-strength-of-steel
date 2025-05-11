'''
========================================================================
This program is a part of personal programming project
Library No: 3
Library_name: PPP_MLR
------------------------------------------------------------------------
Machine learning algorithm: Multiple linear regression(supervised machine learning algorithm for regression)
Adapted the program design of neural network in an attempt to create a single regression model that can perform both MLR and NN
======================================================================== 
'''
# Import required libraries
# -------------------------
import numpy as np
import pandas as pd

# Seed fixed to get known sequence of numbers
np.random.seed(0)
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
        y: Dataframe of dependent feature; Size -> [No of samples x dependent feature(1)]
        ======================================================================== 
        '''
        dataset = pd.read_csv(data)   # Reading in the input dataset
        X = dataset.iloc[:,:-1]       # Selecting all the columns of the input dataset except the last(which is the dependent feature/ target feature)
        y = dataset.iloc[:,[-1]]      # Selecting the last column of the input dataset
        return X,y

    def feature_scaling(self,X,y):
        '''
        ========================================================================
        Description:
        Standardizing the independent and dependent features into a unifrom scale.
        ------------------------------------------------------------------------
        Parameters:
        X: Dataframe of independent features; Size -> [No of samples x independent features]
        y: Dataframe of dependent feature; Size -> [No of samples x dependent feature(1)]
        ------------------------------------------------------------------------
        Check:
        The mean of the standardized data = 0 (close enough)
        The standard deviation of the standardized data = 1 (close enough)
        ------------------------------------------------------------------------
        Return:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        scaled_y: Dataframe of uniformly scaled dependent feature; Size -> [No of samples x dependent feature(1)]
        mean_y: Mean value of the dependent feature; Array size -> [1,1]
        std_y: Standard deviation of the dependent feature; Array size -> [1,1]
        ------------------------------------------------------------------------
        Note:
        -This mean_y and std_y are later used to rescale the dependent feature back into its original scale
        ========================================================================
        '''
        mean_X, std_X = np.array([np.mean(X,axis =0)]), np.array([np.std(X,axis =0)]) # Mean computed row wise for independent features(@axis = 0)
        scaled_X = (X - mean_X)/std_X                                                 # Subtracting the original data from its mean value and dividing by its std.
        mean_y, std_y = np.array([np.mean(y)]), np.array([np.std(y)])
        scaled_y = (y - mean_y)/std_y
        return scaled_X, scaled_y,mean_y,std_y                                          

    def split_dataset(self,X,y):
        '''
        ========================================================================
        Description:
        Randomly splitting the scaled independent and dependent features into training set and testing set, i.e. 80% of the samples are for training set 
        and remaining 20% is for the testing test. 
        ------------------------------------------------------------------------
        Parameters:
        scaled_X: Dataframe of uniformly scaled independent features; Size -> [No of samples x independent features]
        scaled_y: Dataframe of uniformly scaled dependent feature; Size -> [No of samples x dependent feature(1)]
        ------------------------------------------------------------------------
        Return:
        X_train: Training set of independent features with 80% of the samples; Array size -> [80% of samples x independent features]
        X_test: Testing set of independent features with 20% of the samples; Array size -> [20% of samples x independent features]
        y_train: Training set of dependent feature with 80% of the samples; Array size -> [80% of samples x dependent feature(1)]
        y_test: Testing set of dependent feature with 20% of the samples; Array size -> [20% of samples x dependent feature(1)]
        ------------------------------------------------------------------------
        Example:
        scaled_X -> size = [437 x 25]
        After randomly splitting into training set and testing set,
        X_train.shape = (349,25)
        x_test.shape = (88,25)
        ========================================================================
        '''
        X_train, X_test = np.split(X.sample(frac = 1,random_state = 0).values,[int(0.8*len(X))]) # Splitting the scaled independent features randomly
        y_train, y_test = np.split(y.sample(frac = 1,random_state = 0).values,[int(0.8*len(y))]) # Splitting the scaled dependent feature randomly
        return X_train, X_test,y_train,y_test
#
# ======================================================================
# Construction of single layer
# ======================================================================
# 
class Single_layer:
    '''
    ========================================================================
    Description: 
    Performing forward and backward propagation of the single layer.
    ======================================================================== 
    '''

    def __init__(self,inputs,n_neurons):
        '''
        ========================================================================
        Description:
        Initializing weights and biases.
        ------------------------------------------------------------------------
        Parameters:
        inputs: No of independent features; dtype -> int
        n_neurons: No of neurons required in the output layer; dtype -> int
        ------------------------------------------------------------------------
        Outputs:
        weights: Random initialization of weights w.r.t Array of size -> [no of independent features x No of neurons required in the output layer]
        Biases: Set to zeros w.r.t Array of size -> [1 x No of neurons required output layer] 
        ------------------------------------------------------------------------
        Note:
        -Reducing the magnitude of the Gaussian distribution by multiplying it with 0.01 
        -This will reduce the time taken by the model to fit the data during training
        -If the model is non-trainable biases should be assigned with some values 
        ========================================================================
        '''                           
        self.weights = 0.01 * np.random.randn(inputs,n_neurons)    # Creating random weights w.r.t the number of independent features and number of neurons
        self.biases = np.zeros((1,n_neurons))                      # Creating biases w.r.t the number of neurons
    
    def forward_prop(self,inputs):
        '''
        ========================================================================
        Description:
        Creating output layer with known inputs, initialized weights and biases.
        ------------------------------------------------------------------------
        Parameters:
        inputs: Input layer(independent features such as X_train or X_test); Array size -> [No of samples x no of independent features]
        ------------------------------------------------------------------------
        Output:
        output: Output layer; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ========================================================================
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases  # Computing dot product between the inputs and weights and then adding the biases

    def back_prop(self,derivatives):
        '''
        ========================================================================
        Description:
        Finding the derivative inputs of weights,biases and inputs.  
        ------------------------------------------------------------------------
        Parameters:
        derivatives: The backward propagation output of the loss functions will serve as a input here 
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Derivative:
        The derivative of the output w.r.t weights = input -> (1)
        The derivative of the output w.r.t biases = sum(1) -> (2) 
        The derivative of the output w.r.t input = weights -> (3) 
        ------------------------------------------------------------------------
        Output:
        weights_derv: (1) x derivatives; Array size -> [no of independent features x No of neurons required in the output layer]
        biases_derv: (2) x derivatives; Array size -> [1 x No of neurons required in the output layer]
        inputs_derv: (3) x derivatives; Array size -> [No of samples x no of independent features]
        ------------------------------------------------------------------------
        Note:
        -Size of the weights derivative will be equivalent to the size of the weights 
        -Size of the biases derivative will be equivalent to the size of the biases
        -Size of the inputs derivative will be equivalent to the size of the inputs
        ========================================================================
        '''
        self.weights_derv = np.dot(self.inputs.T, derivatives)            # Computing the derivative of the output w.r.t weights
        self.biases_derv = np.sum(derivatives, axis = 0, keepdims = True) # axis = 0 to sum columnwise, keepdim = True is to match the bias dimension
        self.inputs_derv = np.dot(derivatives, self.weights.T)            # Computing the derivative of the output w.r.t input

    def read_weights_biases(self):
        '''
        ========================================================================
        Description:
        Reading in the updated weights and biases after training the model.
        ------------------------------------------------------------------------
        Return:
        weights: Updated weights of the model after training; Array size -> [no of independent features x No of neurons required in the output layer]
        biases: Updated biases of the model after training; Array size -> [1 x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Note:
        -This reading in avoids re-training of the model to similar new datas and allows the model to act 
        as an pure predictor and perform predicitons 
        ========================================================================
        '''
        return self.weights,self.biases
#
# ======================================================================
# Construction of single layer for pure predictor
# ======================================================================
#
class Updated_layer:
    '''
    ========================================================================
    Description:
    Updated layer is same as the single layer it is used when the model act as a pure predictor.
    ======================================================================== 
    '''

    def __init__(self,weights,biases):
        '''
        ========================================================================
        Description:
        Initializing fixed weights and biases. 
        ------------------------------------------------------------------------
        Parameters:
        weights: Reading in the Updated weights of a trained model; Array size -> [no of independent features x No of neurons required in the output layer]
        biases: Reading in the Updated biases of a trained model; Array size -> [1 x No of neurons required in the output layer]
        ========================================================================
        '''                           
        self.weights = weights
        self.biases = biases

    def forward_prop(self,inputs):
        '''
        ========================================================================
        Description:
        Creating output layer with known inputs, fixed weights and biases.
        ------------------------------------------------------------------------
        Parameters:
        inputs: Input layer(scaled independent features); Array size -> [No of samples x no of independent features]
        ------------------------------------------------------------------------
        Output:
        output: Output layer; Array size -> [No of samples  x No of neurons required in the output layer]
        ========================================================================
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases   # Computing dot product between the inputs and weights and then adding the biases
#
# ======================================================================
# Construction of Loss calculation
# ======================================================================
# 
class LossCalculation:
    '''
    ========================================================================
    Description: Computing the loss of the model
    ========================================================================
    '''
    def calculate_loss(self,output, y):
        '''
        ========================================================================
        Description:
        Computing the mean loss of loss per sample.
        ------------------------------------------------------------------------
        Parameters:
        output: The forward propagation output of the single layer(also called as predicted output)
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Test metric(dependent feature such as y_train and y_test); Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        -------------------------------------------------------------------------
        Return:
        loss: loss value of the model w.r.t predicted output and test metric; dtype -> float
        ========================================================================
        ''' 
        lossPer_sample = self.forward_prop(output, y)  # Computing loss per sample using MSE/ RMSE loss
        loss = np.mean(lossPer_sample)                 # Computing the mean loss for all the samples
        return loss
#
# ======================================================================
# Construction of Mean squared error loss
# ======================================================================
#
class MeanSquaredError_loss(LossCalculation):
    '''
    ========================================================================
    Description:
    Performing forward and backward propagation of Mean squared error loss
    ========================================================================
    '''
    def forward_prop(self,y_pred, y):
        '''
        ========================================================================
        Description:
        Computing loss per sample using mean squared error loss w.r.t predicted output and test metric.
        ------------------------------------------------------------------------
        Parameters:
        y_pred: The forward propagation output of the single layer; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Test metric; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Return:
        lossPer_sample: loss value of each sample; Array size -> [No of samples of the input layer]
        ========================================================================
        '''
        lossPer_sample = np.mean((y - y_pred)**2, axis = -1) # Calculating the mean of each sample(if no of neurons > 1) and output it as an vector
        return lossPer_sample
   
    def back_prop(self, derivatives, y):
        '''
        ========================================================================
        Description:
        Finding the derivative input of mean squared error loss.
        ------------------------------------------------------------------------
        Parameters:
        derivatives: The forward propagation output of the single layer
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Test metric(dependent feature y_train); Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Derivative:
        The derivative of the mean squared error loss is taken w.r.t predicted output(y_pred)
        ------------------------------------------------------------------------
        Output:
        inputs_derv: d(mean squared error loss) = -2 * (test metric - derivatives) / no of samples of the input layer 
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ========================================================================
        '''
        n_samples = len(derivatives)                                   # Number of samples(rows)
        outputsPerSample = len(derivatives[0])                         # Number of neurons per sample(colums)
        self.inputs_derv = -2 * (y - derivatives) / outputsPerSample   # Computing the derivative of MSE loss
        self.inputs_derv = self. inputs_derv / n_samples               # Normalizing the computed derivative by dividing it with the number of samples
#
# ======================================================================
# Construction of Root mean squared error loss
# ======================================================================
# 
class RootMeanSquaredError_loss(LossCalculation):
    '''
    ========================================================================
    Description:
    Performing forward and backward propagation of Root mean squared error loss.
    ========================================================================
    '''
    def forward_prop(self,y_pred, y):
        '''
        ========================================================================
        Description:
        Computing loss per sample using root mean squared error loss w.r.t predicted output and test metric.
        ------------------------------------------------------------------------
        Parameters:
        y_pred: The forward propagation output of the single layer; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Test metric; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Return:
        lossPer_sample: loss value of each sample; Array size -> [No of samples of the input layer]
        ========================================================================
        '''
        lossPer_sample = np.sqrt(np.mean((y - y_pred)**2, axis = -1)) # Calculating the mean of each sample(if no of neurons > 1) and output it as an vector
        return lossPer_sample
   
    def back_prop(self, derivatives, y):
        '''
        ========================================================================
        Description:
        Finding the derivative input of root mean squared error loss.
        ------------------------------------------------------------------------
        Parameters:
        derivatives: The forward propagation output of the single layer
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Test metric(dependent feature y_train); Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Derivative:
        The derivative of the root mean squared error loss is taken w.r.t predicted output(y_pred)
        ------------------------------------------------------------------------
        Output:
        inputs_derv: d(root mean squared error loss) = (-1 * (test metric - derivatives) / sqrt((test metric - derivatives)**2) ) / no of samples of the input layer 
        Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ========================================================================
        '''
        n_samples = len(derivatives)                          # Number of samples(rows)
        outputsPerSample = len(derivatives[0])                # Number of neurons per sample(colums)
        self.inputs_derv = (-1 * (y - derivatives) / outputsPerSample) / np.sqrt((y - derivatives)**2 / outputsPerSample)  # Computing the derivative of RMSE loss
        self.inputs_derv = self. inputs_derv / n_samples      # Normalizing the computed derivative by dividing it with the number of samples
#
# ======================================================================
# Construction of Accuracy calculation
# ======================================================================
# 
class Accuracy:
    '''
    ========================================================================
    Description:
    Computing the accuracy of the model
    ========================================================================
    '''
    def coefficient_of_determination(self,y_pred,y):
        '''
        ========================================================================
        Description:
        Computing the coefficient of determination w.r.t predicted output and dependent feature.
        ------------------------------------------------------------------------
        Parameters:
        y_pred: The forward propagation output of the single layer; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        y: Dependent feature such as y_train and y_test; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Return:
        Rsqr: It returns the proportion of variation between the dependent feature and the predicted output of the model; dtype -> float
        ------------------------------------------------------------------------
        Note :
        -The best result is when the predicted values exactly match the dependent feature i.e. R^2 = 1 (close enough)
        ========================================================================
        '''
        SSres = np.sum((y - y_pred)**2)      # Computing the residual sum of squares (1)
        SStot = np.sum((y- np.mean(y))**2)   # Computing the total sum of squares (2)
        Rsqr = 1 - (SSres/SStot)             # Computing Rsqr w.r.t (1) and (2)
        return Rsqr
#
# ======================================================================
# Construction of Stochastic gradient descent Optimizer
# ======================================================================
# 
class SGD_Optimizer:
    '''
    ========================================================================
    Description: Implementing Stochastic gradient descent Optimizer.
    ========================================================================
    '''
    def __init__(self, learning_R = 0.85, learning_R_decay = 0):
        '''
        ========================================================================
        Parameters:
        learning_R: Hyperparameter learning rate; dtype -> float
        learning_R_decay: Hyperparameter learning rate decay; dtype -> float
        ------------------------------------------------------------------------
        Note :
        -if learning_R_decay = 0 then the current learning rate = learning_R and there will be no update
        ========================================================================
        '''
        self.learning_R = learning_R
        self.C_learning_R = learning_R            # Current learning rate
        self.learning_R_decay = learning_R_decay
        self.itr = 0                              # Initially the iteration will be set to 0
   
    def learning_R_update(self):
        '''
        ========================================================================
        Description:
        Update the learning rate after each epoch during training.
        ------------------------------------------------------------------------
        Return:
        C_learning_R: Computing current learning rate using the intial learning rate, learning rate decay and epoch ; dtype -> float
        Current learning rate decay w.r.t epoch     
        ========================================================================
        '''
        if self.learning_R_decay:
            self.C_learning_R = self.learning_R * (1. / (1. +self.learning_R_decay * self.itr)) # Computing current learning rate
            return self.C_learning_R
    
    def parameters_update(self, layer):
        '''
        ========================================================================
        Description:
        Update the weights and biases after each epoch during training.
        ------------------------------------------------------------------------
        Parameters:
        layer: Single layer is given as a input to access the weights and biases of the layers
        ------------------------------------------------------------------------
        Output:
        weights: Updates the weights w.r.t the current learning rate and weights derivatives from backward propagation
        Array size -> [no of independent features x No of neurons required in the output layer]      
        biases: Updates the biases w.r.t the current learning rate and biases derivatives from backward propagation
        Array size -> [1 x No of neurons required in the output layer]
        ========================================================================
        '''
        layer.weights += -self.C_learning_R * layer.weights_derv  # Weights update for each layers with SGD optimizer
        layer.biases += -self.C_learning_R * layer.biases_derv    # Biases update for each layers with SGD optimizer
    
    def itr_update(self):
        '''
        ========================================================================
        Output:
        itr: Equivalent to epoch update during training; dtype -> int      
        ========================================================================
        '''
        self.itr += 1