'''
========================================================================
Test_file: test_ANN_Functions.py
Test_type: Unit test
Aim: To validate activation functions, loss functions and coefficient of determination of a neural network
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
# Construction of dense layer
# ======================================================================
#
class dense_layer:
   '''
   ========================================================================
   Description: 
   Performing forward propagation of dense layers.
   ======================================================================== 
   '''
   def __init__(self,inputs,n_neurons, L2_weight_reg = 0, L2_bias_reg = 0):
      '''
      ========================================================================
      Description:
      Initializing weights, biases and regularization.
      ------------------------------------------------------------------------
      Parameters:
      inputs: No of independent features; dtype -> int
      n_neurons: No of neurons required in the hidden and output layer; dtype -> int
      ------------------------------------------------------------------------
      Outputs:
      weights: Initializing weights to be ones w.r.t Array of size -> [no of independent features x No of neurons required in the hidden and output layer]
      Biases: Set to one; dtype -> int 
      ========================================================================
      '''                           
      self.weights = np.ones((inputs,n_neurons))    
      self.biases = 1                    

      # Assinging regularization
      self.L2_weight_reg = L2_weight_reg
      self.L2_bias_reg = L2_bias_reg
   
   def forward_prop(self,inputs):
      '''
      ========================================================================
      Description:
      Creating hidden and output layer with known inputs, initialized weights and biases.
      ------------------------------------------------------------------------
      Parameters:
      inputs: Fixed inputs read in from the file: parameters.txt and activation function outputs; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Output:
      output: Hidden and output layer; Array size -> [No of samples x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases
      return self.output
#
# ======================================================================
# Construction of Rectified Linear Units(ReLU) activation function
# ======================================================================
# 
class ReLU_Activation:
   def forward_prop(self,inputs):
      '''
      ========================================================================
      Description:
      Replacing the negative values in the hidden layer by zeros.
      ------------------------------------------------------------------------
      Parameters:
      inputs: Hidden layer; Array size -> [No of samples x No of neurons required in the hidden layer]
      ------------------------------------------------------------------------
      Output:
      output: Negative values in the hidden layer are replaced by zeros; Array size -> [No of samples x No of neurons required in the hidden layer]
      ------------------------------------------------------------------------
      Note:
      -ReLU activation function is used in the hidden layers and the size of the inputs and output are the same
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.maximum(0, inputs)    
      return self.output
#
# ======================================================================
# Construction of Linear activation function
# ======================================================================
#
class Linear_Activation:
   '''
   ========================================================================
   Description:
   Performing forward propagation of Linear activation function.
   ======================================================================== 
   '''
   def forward_prop(self, inputs):
      '''
      ========================================================================
      Description:
      Due to the linear nature of the function(y = x) the input will be the output.
      ------------------------------------------------------------------------
      Parameters:
      inputs: output layer; Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Output:
      output: predicted output; Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Note:
      -Linear activation function is used in the output layer
      ========================================================================
      '''
      self.inputs = inputs
      self.output = inputs
      return self.output
#
# ======================================================================
# Construction of Mean squared error loss
# ======================================================================
# 
class MeanSquaredError_loss:
   '''
   ========================================================================
   Description:
   Performing forward propagation of Mean squared error loss.
   ========================================================================
   '''
   def forward_prop(self,y_pred, y):
      '''
      ========================================================================
      Description:
      Computing loss using mean squared error loss w.r.t predicted output and test metric.
      ------------------------------------------------------------------------
      Parameters:
      y_pred: The forward propagation output of the linear activation functions; Array size -> [No of samples  x No of neurons required in the output layer]
      y: Test metric; Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Return:
      loss: loss value of the sample; dtype -> float
      ========================================================================
      '''
      lossPer_sample = np.mean((y - y_pred)**2, axis = -1) # Calculating mean of each row and outputting it as an vector
      loss = np.mean(lossPer_sample)
      return loss
   
   def regularization_loss(self,layer):
      '''
      ========================================================================
      Description:
      Computing regularization loss from weights and biases of each dense layer.
      ------------------------------------------------------------------------
      Parameters:
      layer: dense layers are given as inputs and are used to access the weights and biases of the layers
      ------------------------------------------------------------------------
      Return:
      reg_loss: Regularization loss at weights and biases of a dense layer; dtype -> float 
      ------------------------------------------------------------------------
      Note :
      -L2 Regularization are used to reduce the neurons attempt to memorizing a data element 
      ========================================================================
      '''
      reg_loss = 0 # initially 0

      # Regularization at weights 
      if layer.L2_weight_reg > 0:
         reg_loss += layer.L2_weight_reg  * np.sum(layer.weights * layer.weights)
        
      # Regularization at biases 
      if layer.L2_bias_reg > 0:
         reg_loss += layer.L2_bias_reg  * np.sum(layer.biases * layer.biases)
         
      return reg_loss

def pathto_parameters():
   '''
   ========================================================================
   Description:
   Setting path to open parameters file from different directories
   ------------------------------------------------------------------------
   Return:
   filename: Path to the parameters file 
   ========================================================================
   '''
   cwd = os.path.basename(os.getcwd())                            # Locating current working directory
   if cwd == 'test_ANN_Functions':                                # If tested from the current directory
      filename = 'parameters.txt'
   elif cwd == 'test_ANN':                                        # If tested from test_ANN
      open_path = os.path.abspath('test_ANN_Functions')        
      filename = os.path.join(open_path,'parameters.txt')
   elif cwd == 'test_ML':                                         # If tested from test_ML
      open_path = os.path.abspath('test_ANN/test_ANN_Functions')           
      filename = os.path.join(open_path,'parameters.txt')
   else:                                                          # Else an error will be thrown stating that the user is in the wrong directory
      sys.exit('Error:Testing executed in the wrong directory\nPossible directories: test_ML, test_ANN and current directory of the file')
   return filename   

def read_Activation_inputs(start,end):
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set and expected results for two test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   start: Starting line of the parameter set
   end: Last line of the parameter set
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set and expected are fixed and should not be modified
   ========================================================================
   '''
   input = []                                                        # Creating storage to store input set
   expRes = []                                                       # Creating storage to store expected results
   
   filename = pathto_parameters()                                    # File path to open the parameters file
   with open(filename, 'r') as f:                                    # Read parameters
      read_param = f.readlines()
      for idx in np.arange(start,end,1):                             # Iterating over the fixed no of lines
         read_input = read_param[idx].split(' ')[0]                  # Reading in the fixed no of input set
         read_expectedRes = read_param[idx].split(' ')[2]            # Reading in the fixed no of expected result
         input.append(list(map(int,read_input.split(','))))          # Storing the input set
         expRes.append(list(map(int,read_expectedRes.split(','))))   # Storing the expected result

   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array(input[0]), np.array(expRes[0])),(np.array(input[1]), np.array(expRes[1]))]
   return param

def read_Loss_inputs(start,end):
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set, prediction and expected results for two test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   start: Starting line of the parameter set
   end: Last line of the parameter set
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set, prediction and expected are fixed and should not be modified
   ========================================================================
   '''
   input1 = []                                                       # Creating storage to store input set for test case 1
   input2 = []                                                       # Creating storage to store input set for test case 2
   predRes = []                                                      # Creating storage to store prediction results
   expRes = []                                                       # Creating storage to store expected results
   
   filename = pathto_parameters()                                    # File path to open the parameters file
   with open(filename, 'r') as f:                                    # Read parameters
      read_param = f.readlines()
      for idx in np.arange(start,end,1):                             # Iterating over the fixed no of lines
         read_input1 = read_param[idx].split(' ')[0]                 # Reading in the fixed no of input set for test case 1
         read_input2 = read_param[idx].split(' ')[2]                 # Reading in the fixed no of input set for test case 2
         read_predRes = read_param[idx].split(' ')[4]                # Reading in the fixed no of prediciton results
         read_expectedRes = read_param[idx].split(' ')[6]            # Reading in the fixed no of expected result

         input1.append(list(map(int,read_input1.split(','))))        # Storing the input set of test case 1
         input2.append(list(map(int,read_input2.split(','))))        # Storing the input set of test case 1
         predRes.append(list(map(int,read_predRes.split(','))))      # Storing the prediction results
         expRes.append(list(map(int,read_expectedRes.split(','))))   # Storing the expected results
      
   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array([input1[0],input1[1],input1[2]]),np.ones((predRes[0][0],predRes[0][1])),np.array(expRes[0])),(np.array([input2[0],input2[1],input2[2]])\
      ,np.ones((predRes[1][0],predRes[1][1])),np.array(expRes[1]))] 
   return param

def read_Regloss_inputs(start,end):
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set, dependent feature and expected loss results for two test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   start: Starting line of the parameter set
   end: Last line of the parameter set
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set, dependent feature and expected loss are fixed and should not be modified
   ========================================================================
   '''
   input = []                                                       # Creating storage to store input set 
   depFet = []                                                      # Creating storage to store dependent feature
   exploss = []                                                     # Creating storage to store expected loss
   expRegloss = []                                                  # Creating storage to store expected regularization loss
   
   filename = pathto_parameters()                                   # File path to open the parameters file
   with open(filename, 'r') as f:                                   # Read parameters
      read_param = f.readlines()
      for idx in np.arange(start,end,1):                            # Iterating over the fixed no of lines
         read_input = read_param[idx].split(' ')[0]                 # Reading in the fixed no of input set
         read_depFet = read_param[idx].split(' ')[2]                # Reading in the fixed dependent feature
         read_exploss = read_param[idx].split(' ')[4]               # Reading in the fixed expected loss
         read_expRegloss = read_param[idx].split(' ')[6]            # Reading in the fixed expected regularization loss

         input.append(list(map(int,read_input.split(','))))         # Storing the input set 
         depFet.append(float(read_depFet))                          # Storing the dependent feature
         exploss.append(float(read_exploss))                        # Storing the expected loss
         expRegloss.append(float(read_expRegloss))                  # Storing the expected regularization loss
   
   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array([input[0]]), depFet[0], exploss[0], expRegloss[0]), (np.array([input[1]]), depFet[1], exploss[1],expRegloss[1])]
   return param

def read_CoefDet_inputs(start,end):
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set, prediction and expected results for two test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   start: Starting line of the parameter set
   end: Last line of the parameter set
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set, prediction and expected are fixed and should not be modified
   ========================================================================
   '''
   input1 = []                                                       # Creating storage to store input set for test case 1
   input2 = []                                                       # Creating storage to store input set for test case 2
   predRes1 = []                                                     # Creating storage to store prediction results for test case 1
   predRes2 = []                                                     # Creating storage to store prediction results for test case 2
   expRes = []                                                       # Creating storage to store expected results
   
   filename = pathto_parameters()                                    # File path to open the parameters file
   with open(filename, 'r') as f:                                    # Read parameters
      read_param = f.readlines()
      for idx in np.arange(start,end,1):                             # Iterating over the fixed no of lines
         read_input1 = read_param[idx].split(' ')[0]                 # Reading in the fixed no of input set for test case 1
         read_input2 = read_param[idx].split(' ')[2]                 # Reading in the fixed no of input set for test case 2
         read_predRes1 = read_param[idx].split(' ')[4]               # Reading in the fixed no of prediciton results for test case 1
         read_predRes2 = read_param[idx].split(' ')[6]               # Reading in the fixed no of prediciton results for test case 2
         read_expectedRes = read_param[idx].split(' ')[8]            # Reading in the fixed no of expected result

         input1.append(list(map(int,read_input1.split(','))))        # Storing the input set of test case 1
         input2.append(list(map(int,read_input2.split(','))))        # Storing the input set of test case 2
         predRes1.append(list(map(int,read_predRes1.split(','))))    # Storing the prediction results of test case 1
         predRes2.append(list(map(float,read_predRes2.split(','))))  # Storing the prediction results of test case 2
         expRes.append(float(read_expectedRes))                      # Storing the expected results
 
   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array([input1[0],input1[1],input1[2]]),np.array([predRes1[0],predRes1[1],predRes1[2]]),expRes[0]),(np.array([input2[0],input2[1],input2[2]])\
   ,np.array([predRes2[0],predRes2[1],predRes2[2]]),expRes[1])]

   return param

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,expected', read_Activation_inputs(9,11))

def test_ReLU_Activation(input,expected):
   '''
   ========================================================================
   Description:
   Testing ReLU activation function
   ------------------------------------------------------------------------
   Parameters:
   input: Input parameters; Array size -> [n x n]
   expected: Expected results; Array size -> [n x n]
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   activation_function = ReLU_Activation()
   output = activation_function.forward_prop(input)
   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(output ,expected), 'test failed'

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,expected', read_Activation_inputs(15,17))

def test_Linear_Activation(input,expected):
   '''
   ========================================================================
   Description:
   Testing Linear activation function
   ------------------------------------------------------------------------
   Parameters:
   input: Input parameters; Array size -> [m x n]
   expected: Expected results; Array size -> [m x n]
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   activation_function = Linear_Activation()
   output = activation_function.forward_prop(input)
   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(output ,expected), 'test failed'

# Defining multiple test cases for testing
#------------------------------------------------------------------------   
@pytest.mark.parametrize('output, prediction,expected', read_Loss_inputs(21,24)) # Considering output with three neurons per sample

def test_MeanSquaredError_loss(output, prediction,expected):
   '''
   ========================================================================
   Description:
   Testing Mean squared error loss function
   ------------------------------------------------------------------------
   Parameters:
   output: Input parameters for test case 1 & 2; Array size -> [m x n]
   prediction: Prediction results; Array size -> [m x n]
   expected: Expected results; Array size -> [1 x m]
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   lossPer_sample = np.mean((output - prediction)**2, axis = -1)                 # Calculating mean of each row and outputting it as an vector
   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(lossPer_sample ,expected), 'test failed' 

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('output, prediction,expected',  read_Loss_inputs(28,31)) # Considering output with three neurons per sample

def test_RootMeanSquaredError_loss(output, prediction,expected):
   '''
   ========================================================================
   Description:
   Testing Root mean squared error loss function
   ------------------------------------------------------------------------
   Parameters:
   output: Input parameters for test case 1 & 2; Array size -> [m x n]
   prediction: Prediction results; Array size -> [m x n]
   expected: Expected results; Array size -> [1 x m]
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   lossPer_sample = np.sqrt(np.mean((output - prediction)**2, axis = -1))
   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(lossPer_sample ,expected), 'test failed'

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,y,exp_ls,exp_regls',read_Regloss_inputs(35,37))
                                             
def test_Regularization_loss(input,y,exp_ls,exp_regls):
   '''
   ========================================================================
   Description:
   Testing Regularization loss
   ------------------------------------------------------------------------
   Parameters:
   input: Input parameters; Array size -> [1 x 5]
   y: Dependent feature w.r.t the input set; dtype -> float
   exp_ls: Expected loss; dtype -> float
   exp_regls: Expected regularization loss; dtype -> float
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   # Initialization for no of layers, activation functions and loss function
   #------------------------------------------------------------------------
   dense_layer_1 =dense_layer(5,3,2e-6,2e-6)  # Assigning fixed regularization inputs
   activation_function_1 = ReLU_Activation()
   dense_layer_2 = dense_layer(3,3,3e-6,3e-6) # Assigning fixed regularization inputs
   activation_function_2 = ReLU_Activation()
   dense_layer_3 = dense_layer(3,1) 
   activation_function_3 = Linear_Activation()
   loss_function = MeanSquaredError_loss()

   # Begin forward propagation
   #------------------------------------------------------------------------
   dense_layer_1.forward_prop(input) 
   activation_function_1.forward_prop(dense_layer_1.output) 
   dense_layer_2.forward_prop(activation_function_1.output) 
   activation_function_2.forward_prop(dense_layer_2.output)  
   dense_layer_3.forward_prop(activation_function_2.output)
   y_pred = activation_function_3.forward_prop(dense_layer_3.output)

   # Compute loss
   #------------------------------------------------------------------------
   loss = loss_function.forward_prop(y,y_pred)
   reg_loss = loss_function.regularization_loss(dense_layer_1) + loss_function.regularization_loss(dense_layer_2)  + loss_function.regularization_loss(dense_layer_3)
   total_loss = loss + reg_loss
   exp_totls = exp_ls + exp_regls

   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(loss ,exp_ls), 'test failed'              # Checking loss
   np.testing.assert_array_almost_equal(reg_loss ,exp_regls), 'test failed'       # Checking regularization loss
   np.testing.assert_array_almost_equal(total_loss ,exp_totls), 'test failed'     # Checking total attained loss(loss + reg loss)

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('output, prediction,expected', read_CoefDet_inputs(41,44))

def test_coefficient_of_determination(output, prediction,expected):
   '''
   ========================================================================
   Description:
   Testing Coefficient of determination
   ------------------------------------------------------------------------
   Parameters:
   output: Input parameters for test case 1 & 2; Array size -> [m x n]
   prediction: Prediction results for test case 1 & 2; Array size -> [m x n]
   expected: Expected results is a scalar; dtype -> int or float
   ------------------------------------------------------------------------
   Return:
   Assertion result
   ========================================================================
   '''
   SSres = np.sum((prediction - output)**2)
   SStot = np.sum((prediction- np.mean(prediction))**2)
   Rsqr = 1 - (SSres/SStot)
   
   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(Rsqr ,expected), 'test failed'