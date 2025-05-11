'''
========================================================================
Test_file: test_ANN_Forward_prop
Test_type: Unit test
Aim: To validate one complete forward propagation step of a neural network
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
class Dense_layer:
   '''
   ========================================================================
   Description: 
   Performing forward propagation of dense layers.
   ======================================================================== 
   '''
   def __init__(self,inputs,n_neurons):
      '''
      ========================================================================
      Description:
      Initializing weights, biases.
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

   def forward_prop(self,inputs):
      '''
      ========================================================================
      Description:
      Creating hidden and output layer with known inputs, initialized weights and biases.
      ------------------------------------------------------------------------
      Parameters:
      inputs: Fixed inputs are read in from the file: parameters.txt and activation function outputs; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Output:
      output: Hidden and output layer; Array size -> [No of samples x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases
#
# ======================================================================
# Construction of Rectified Linear Units(ReLU) activation function
# ======================================================================
# 
class ReLU_Activation:
   '''
   ========================================================================
   Description:
   Performing forward propagation of ReLU activation function
   ======================================================================== 
   '''
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

def read_parameters():
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set and expected results for five test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   inputset: Input parameters; Array size -> [1 x 5]
   expected: Expected results for each test case; dtype -> int
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set and expected are fixed and should not be modified
   ========================================================================
   '''
   input = []                                                     # Creating storage to store input set
   expRes = []                                                    # Creating storage to store expected results

   # Setting path to open parameters file
   #------------------------------------------------------------------------
   cwd = os.path.basename(os.getcwd())                            # Locating current working directory
   if cwd == 'test_ANN_Forward_prop':                             # If tested from the current directory
      filename = 'parameters.txt'
   elif cwd == 'test_ANN':                                        # If tested from test_ANN
      open_path = os.path.abspath('test_ANN_Forward_prop')        
      filename = os.path.join(open_path,'parameters.txt')
   elif cwd == 'test_ML':                                         # If tested from test_ML
      open_path = os.path.abspath('test_ANN/test_ANN_Forward_prop')           
      filename = os.path.join(open_path,'parameters.txt')
   else:                                                          # Else an error will be thrown stating that the user is in the wrong directory
      sys.exit('Error:Testing executed in the wrong directory\nPossible directories: test_ML, test_ANN and current directory of the file')
      
   with open(filename, 'r') as f:                                 # Read parameters
      read_param = f.readlines()
      for idx in np.arange(9,14,1):                               # Iterating over the fixed no of lines
         read_input = read_param[idx].split(' ')[0]               # Reading in the fixed no of input set
         read_expectedRes = read_param[idx].split(' ')[2]         # Reading in the fixed no of expected result
         input.append(list(map(int,read_input.split(','))))       # Storing the input set
         expRes.append(int( read_expectedRes))                    # Storing the expected results
         
   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array(input[0]), expRes[0]),(np.array(input[1]), expRes[1]),(np.array(input[2]), expRes[2]),(np.array(input[3]), expRes[3]),(np.array(input[4]), expRes[4])]
   return param
  
# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,expected', read_parameters())

def test_CompleteForward_prop(input,expected):
   '''
   ========================================================================
   Description:
   Test one complete forward propagation of the neural network 
   ------------------------------------------------------------------------
   Parameters:
   input: Input parameters for each test case; Array size -> [1 x 5]
   expected: Expected results for each test case; dtype -> int
   ------------------------------------------------------------------------
   Note:
   -The number of neurons in both the hidden layers are fixed to three
   ========================================================================
   '''
   # Initialization  for no of layers and activation functions
   #------------------------------------------------------------------------
   dense_layer_1 =Dense_layer(5,3)                                    
   activation_function_1 = ReLU_Activation()
   dense_layer_2 = Dense_layer(3,3) 
   activation_function_2 = ReLU_Activation()
   dense_layer_3 = Dense_layer(3,1) 
   activation_function_3 = Linear_Activation()
   
   # Begin forward propagation
   #------------------------------------------------------------------------
   dense_layer_1.forward_prop(input) 
   activation_function_1.forward_prop(dense_layer_1.output) 
   dense_layer_2.forward_prop(activation_function_1.output) 
   activation_function_2.forward_prop(dense_layer_2.output)  
   dense_layer_3.forward_prop(activation_function_2.output)
   activation_function_3.forward_prop(dense_layer_3.output)

   # Check if the expected result and the obtained result are the same
   #------------------------------------------------------------------------
   assert activation_function_3.output == expected, 'test failed'






