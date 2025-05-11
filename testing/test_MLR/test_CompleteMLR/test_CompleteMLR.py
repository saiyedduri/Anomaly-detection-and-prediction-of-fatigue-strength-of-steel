'''
========================================================================
Test_file: test_CompleteMLR.py
Test_type: Gradient checking & Unit test
Aim: To validate one complete training step using SGD optimizer
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
        Output:
        weights: Initializing weights to be ones w.r.t Array of size -> [no of independent features x No of neurons required in the hidden and output layer]
        Biases: Set to one; dtype -> int 
        ========================================================================
        '''
        self.weights = np.ones((inputs,n_neurons))    
        self.biases = 1

    def forward_prop(self,inputs):
        '''
        ========================================================================
        Description:
        Creating output layer with known inputs, initialized weights and biases.
        ------------------------------------------------------------------------
        Parameters:
        inputs: Fixed inputs are read in from the file: parameters.txt; Array size -> [No of samples x no of independent features]
        ------------------------------------------------------------------------
        Output:
        output: Output layer; Array size -> [No of samples x No of neurons required in the output layer]
        ========================================================================
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def back_prop(self,derivatives):
        '''
        ========================================================================
        Description:
        Finding the derivative inputs of weights,biases and inputs.  
        ------------------------------------------------------------------------
        Parameters:
        derivatives: The backward propagation output of the loss functions will serve as a input here 
        Array size -> [No of samples x No of neurons required in the output layer]
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
        ========================================================================
        '''
        self.weights_derv = np.dot(self.inputs.T, derivatives)
        self.biases_derv = np.sum(derivatives, axis = 0, keepdims = True) 
        self.inputs_derv = np.dot(derivatives, self.weights.T)
        return self.weights_derv, self.biases_derv, self.inputs_derv
#
# ======================================================================
# Construction of Mean squared error loss
# ======================================================================
#
class MeanSquaredError_loss:
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
        y_pred: The forward propagation output of the single layer; Array size -> [No of samples x No of neurons required in the output layer]
        y: Test metric; Array size -> [No of samples x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Return:
        lossPer_sample: loss value of each sample; Array size -> [No of samples of the input layer]
        ========================================================================
        '''
        lossPer_sample = np.mean((y - y_pred)**2, axis = -1)
        loss = np.mean(lossPer_sample) 
        return loss
   
    def back_prop(self, derivatives, y): 
        '''
        ========================================================================
        Description:
        Finding the derivative input of mean squared error loss.
        ------------------------------------------------------------------------
        Parameters:
        derivatives: The forward propagation output of the single layer
        Array size -> [No of samples x No of neurons required in the output layer]
        y: Test metric(dependent feature y_train); Array size -> [No of samples x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Derivative:
        The derivative of the mean squared error loss is taken w.r.t predicted output(y_pred)
        ------------------------------------------------------------------------
        Output:
        inputs_derv: d(mean squared error loss) = -2 * (test metric - derivatives) / no of samples 
        Array size -> [No of samples x No of neurons required in the output layer]
        ========================================================================
        '''
        n_samples = 1 # len(derivatives) # Number of samples(rows)   
        outputsPerSample = 1 # len(derivatives[0]) # Number of neurons per sample(colums)
        self.inputs_derv = -2 * (y - derivatives) / outputsPerSample
        self.inputs_derv = self. inputs_derv / n_samples
        return self.inputs_derv
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
        self.C_learning_R = learning_R                # Current learning rate
        self.learning_R_decay = learning_R_decay
        self.itr = 1                                  # Generally itr = 0; As we want the result of single step itr = 1
   
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
            self.C_learning_R = self.learning_R * (1. / (1. +self.learning_R_decay * self.itr))
            return self.C_learning_R
    
    def parameters_update(self, layer):
        '''
        ========================================================================
        Description:
        Update the weights and biases after each epoch during training.
        ------------------------------------------------------------------------
        Parameters:
        layer: dense layers are given as inputs and are used to access the weights and biases of the layers
        ------------------------------------------------------------------------
        Output:
        weights: Updates the weights w.r.t the current learning rate and weights derivatives from backward propagation
        Array size -> [no of independent features x No of neurons required in the output layer]      
        biases: Updates the biases w.r.t the current learning rate and biases derivatives from backward propagation
        Array size -> [1 x No of neurons required in the output layer]
        ========================================================================
        '''
        layer.weights += -self.C_learning_R * layer.weights_derv
        layer.biases += -self.C_learning_R * layer.biases_derv
    
    def itr_update(self):
        '''
        ========================================================================
        Output:
        itr: Equivalent to epoch update during training; dtype -> int      
        ========================================================================
        '''
        self.itr += 1
#
# ======================================================================
# Gradient checking of back propagation
# ======================================================================
#
def gradient_checking(plusepsilon, minusepsilon, epsilon):
   '''
   ========================================================================
   Description:
   Testing the neural network using numerical approximation of the derivative (gradient checking)
   ------------------------------------------------------------------------
   Parameters:
   plusepsilon: Input to the function plus epsilon
   minusepsilon: Input to the function minus epsilon
   epsilon: Small pertubation of 0.001; dtype -> float
   ------------------------------------------------------------------------
   Return:
   GradCheck_BP: Numerical gradient result
   ========================================================================
   '''
   GradCheck_BP = (plusepsilon - minusepsilon)/(2*epsilon)
   return GradCheck_BP

def back_prop_lossfunc(loss_function, output, y , epsilon):
    '''
    ========================================================================
    Description:
    Performing backward propagation step for loss function and validating the results with gradient checking
    ------------------------------------------------------------------------
    Parameters:
    loss_function: Loss function to be used for back propagation
    output: Training results
    y: Dependent feature
    epsilon: Small pertubation of 0.001; dtype -> float
    ------------------------------------------------------------------------
    Return:
    Assertion result 
    ========================================================================
    '''    
    loss_back_prop = loss_function.back_prop(output, y) # Analytical solution

    plusepsilon = loss_function.forward_prop(output + epsilon,y)
    minusepsilon = loss_function.forward_prop(output - epsilon,y)
    
    # Gradient check
    #------------------------------------------------------------------------
    GradCheck_BP_loss = gradient_checking(plusepsilon, minusepsilon,epsilon)

    # Assertion
    #------------------------------------------------------------------------
    np.testing.assert_array_almost_equal(loss_back_prop,GradCheck_BP_loss), 'test failed'

def back_prop_Singlelayer(single_layer,derivative,input,epsilon):
    '''
    ========================================================================
    Description:
    Performing backward propagation step for activation function and validating the results with gradient checking
    ------------------------------------------------------------------------
    Parameters:
    Single_layer: Forward propagation input
    derivative : Derivative input of chain rule
    input: Input dataset
    epsilon: Small perturbation of 0.001; dtype -> float
    ------------------------------------------------------------------------
    Return:
    Assertion results for weights, biases and inputs
    And numerical gradient of weights and biases
    ========================================================================
    '''
    weightsBP, biasesBP, inputsBP = single_layer.back_prop(derivative.inputs_derv) # Analytical solution
    
    # Forward propagation step of dense layers added with pertubation and multiplied with input derivative because of chain rule
    #------------------------------------------------------------------------
    plusepsilon_weightsBP = (np.dot(input.T,derivative.inputs_derv) * (single_layer.weights + epsilon)) + single_layer.biases 
    minuspsilon_weightsBP = (np.dot(input.T,derivative.inputs_derv)  * (single_layer.weights - epsilon)) + single_layer.biases
    plusepsilon_biasesBP = np.dot(input ,single_layer.weights) + (single_layer.biases + epsilon) * derivative.inputs_derv
    minuspsilon_biasesBP = np.dot(input ,single_layer.weights) + (single_layer.biases - epsilon) * derivative.inputs_derv
    plusepsilon_inputsBP = ((input + epsilon) * np.dot(derivative.inputs_derv,single_layer.weights.T)) + single_layer.biases
    minuspsilon_inputsBP = ((input - epsilon) * np.dot(derivative.inputs_derv,single_layer.weights.T)) + single_layer.biases

    # Gradient check with chain rule
    #------------------------------------------------------------------------
    GradCheck_weightsBP = gradient_checking(plusepsilon_weightsBP, minuspsilon_weightsBP,epsilon) 
    GradCheck_biasesBP = gradient_checking(plusepsilon_biasesBP, minuspsilon_biasesBP,epsilon) 
    GradCheck_inputsBP = gradient_checking(plusepsilon_inputsBP, minuspsilon_inputsBP,epsilon) 
    # Assertion
    #------------------------------------------------------------------------
    np.testing.assert_array_almost_equal(weightsBP ,GradCheck_weightsBP), 'test failed'
    np.testing.assert_array_almost_equal(biasesBP, GradCheck_biasesBP), 'test failed'
    np.testing.assert_array_almost_equal(inputsBP ,GradCheck_inputsBP), 'test failed'

    return GradCheck_weightsBP, GradCheck_biasesBP

def optimizer_learningRate(optimizer,ExpSGD_Current_LR):
    '''
    ========================================================================
    Description:
    Validating the expected current learning rate with obtained one
    ------------------------------------------------------------------------
    Parameters:
    optimizer: Optimizer used -> SGD
    ExpSGD_Current_LR: Expected learning rate; dtype -> float
    ------------------------------------------------------------------------
    Return:
    Assertion results for current learning rate
    ========================================================================
    '''
    SGD_Current_LR = optimizer.learning_R_update()  # Analytical solution

    # Assertion
    #------------------------------------------------------------------------
    assert abs(SGD_Current_LR - ExpSGD_Current_LR) < 0.0001, 'test failed' 

def optimizer_SGD(optimizer,Single_layer,ExpSGD_Current_LR,GradCheck_weightsBP,GradCheck_biasesBP):
    '''
   ========================================================================
   Description:
   Performing optimization step and validating the results by computing optimization using numerical gradient results
   ------------------------------------------------------------------------
   Parameters:
   optimizer: Optimizer used -> SGD
   Single_layer: Single layer input
   ExpSGD_Current_LR: Expected learning rate; dtype -> float
   GradCheck_weightsBP: Weight derivative computed by numerical gradient
   GradCheck_biasesBP: Bias derivative computed by numerical gradient
   ------------------------------------------------------------------------
   Return:
   Assertion results for the parameters update
   ========================================================================
   '''
    # Compute optimization
    #------------------------------------------------------------------------
    optimizer.parameters_update(Single_layer)
    SGD_SL_Weights,SGD_SL_biases = Single_layer.weights, Single_layer.biases

    # Optimization computed using numerical gradient results
    #------------------------------------------------------------------------
    ExpSGD_SL_Weights = np.ones((Single_layer.weights.shape)) -ExpSGD_Current_LR * GradCheck_weightsBP
    ExpSGD_SL_biases = 1 -ExpSGD_Current_LR * GradCheck_biasesBP

    # Assertion
    #------------------------------------------------------------------------
    np.testing.assert_array_almost_equal(SGD_SL_Weights ,ExpSGD_SL_Weights), 'test failed'
    np.testing.assert_array_almost_equal(SGD_SL_biases ,ExpSGD_SL_biases), 'test failed'

def read_parameters():
    '''
    ========================================================================
    Description:
    Reading in parameters such as input set and dependent feature for five test cases from the file: parameters.txt in the same directory 
    ------------------------------------------------------------------------
    Parameters:
    inputset: Input parameters; Array size -> [1 x 5]
    dependent feature: Dependent feature for each test case; dtype -> float
    ------------------------------------------------------------------------
    Return:
    param: Input set for pytest.mark.parametrize to check and varify all the test cases
    ------------------------------------------------------------------------
    Note:
    -The number of entries for the input set, dependent feature and expected forward propagation are fixed and should not be modified
    ========================================================================
    '''
    input = []                                                      # Creating storage to store input set
    depFeat = []                                                    # Creating storage to store dependent feature
    expFP = []                                                      # Creating storage to store expected forward propagation results

    # Setting path to open parameters file
    #------------------------------------------------------------------------
    cwd = os.path.basename(os.getcwd())                             # Locating current working directory
    if cwd == 'test_CompleteMLR':                                   # If tested from the current directory
        filename = 'parameters.txt'
    elif cwd == 'test_MLR':                                         # If tested from test_MLR
        open_path = os.path.abspath('test_CompleteMLR')           
        filename = os.path.join(open_path,'parameters.txt')
    elif cwd == 'test_ML':                                          # If tested from test_ML
        open_path = os.path.abspath('test_MLR/test_CompleteMLR')           
        filename = os.path.join(open_path,'parameters.txt')
    else:                                                           # Else an error will be thrown stating that the user is in the wrong directory
        sys.exit('Error:Testing executed in the wrong directory\nPossible directories: test_ML, test_ANN and current directory of the file')
        
    with open(filename, 'r') as f:                                  # Read parameters
        read_param = f.readlines()
        for idx in np.arange(8,13,1):                               # Iterating over the fixed no of lines
            read_input = read_param[idx].split(' ')[0]              # Reading in the fixed no of input set
            read_depFeat = read_param[idx].split(' ')[2]            # Reading in the fixed no of dependent feature
            read_expFPRes = read_param[idx].split(' ')[4]           # Reading in the fixed no of expected result
            input.append(list(map(int,read_input.split(','))))      # Storing the input set
            depFeat.append(float(read_depFeat))                     # Storing the dependent feature
            expFP.append(int(read_expFPRes))                        # Storing expected forward propagation results

    # Creating test cases for parametrize
    #------------------------------------------------------------------------
    param = [(np.array([input[0]]), depFeat[0],expFP[0],0.001,0.772727272),(np.array([input[1]]),depFeat[1],expFP[1], 0.001,0.772727272),(np.array([input[2]]),depFeat[2],\
    expFP[2],0.001,0.772727272),(np.array([input[3]]),depFeat[3],expFP[3],0.001,0.772727272),(np.array([input[4]]), depFeat[4],expFP[4],0.001,0.772727272)]
    return param

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,y,Exp_FP,epsilon,Exp_LR',read_parameters())

def test_CompleteMLR(input,y,Exp_FP,epsilon,Exp_LR):
    '''
    ========================================================================
    Description:
    Test one complete operation of the neural network with SGD optimizer by using numerical gradient
    ------------------------------------------------------------------------
    Parameters:
    input: Input parameters for each test case; Array size -> [1 x 5]
    y: Dependent feature for each test case; dtype -> float
    Exp_FP: Expected forward propagation input; dtype -> int
    epsilon: Small pertubation of 0.001; dtype -> float
    Exp_LR: Expected learning rate; dtype -> float
    ------------------------------------------------------------------------
    Note:
    -The number of neurons in the input and output layers are fixed
    -The specified hyperparameters are also fixed
    ========================================================================
    '''
    # Initialization  for single layers, loss function and optimizer
    #-----------------------------------------------------------------------
    single_layer = Single_layer(5,1)  
    loss_function = MeanSquaredError_loss()
    optimizer = SGD_Optimizer(learning_R = 0.85, learning_R_decay = 1e-1)

    # Begin forward propagation
    #-----------------------------------------------------------------------
    single_layer.forward_prop(input)

    # Assertion
    #------------------------------------------------------------------------
    np.testing.assert_array_almost_equal(single_layer.output ,Exp_FP), 'test failed' 

    # Test for first back propagation step : loss back prop
    back_prop_lossfunc(loss_function, single_layer.output, y , epsilon)

    # Test for last back propagation step : back prop of Single layer 
    GC_weightsSL,GC_biasesSL= back_prop_Singlelayer(single_layer,loss_function,input,epsilon)
    
    # Test learning rate
    optimizer_learningRate(optimizer,Exp_LR)

    # Checking SGD Optimizer for parameters updates in Single layer 
    optimizer_SGD(optimizer,single_layer,Exp_LR,GC_weightsSL,GC_biasesSL)
   