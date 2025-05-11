'''
========================================================================
Test_file: test_ANN_Back_prop.py
Test_type: Gradient checking
Aim: To validate one complete backward propagation step of a neural network
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
   Performing forward and backward propagation of dense layers.
   ======================================================================== 
   '''
   def __init__(self,inputs,n_neurons, L2_weight_reg = 0, L2_bias_reg = 0):
      '''
      ========================================================================
      Description:
      Initializing weights,biases and regularization.
      ------------------------------------------------------------------------
      Parameters:
      inputs: No of independent features; dtype -> int
      n_neurons: No of neurons required in the hidden and output layer; dtype -> int
      L2_weight_reg: lamda hyperparameter @ weights; dtype -> float
      L2_bias_reg: lamda hyperparameter @ biases; dtype -> float
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
      inputs: Fixed inputs are read in from the file: parameters.txt and activation function outputs; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Output:
      output: Hidden and output layer; Array size -> [No of samples x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases
      return self.output

   def back_prop(self,derivatives):
      '''
      ========================================================================
      Description:
      Finding the derivative inputs of weights,biases and inputs.
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The backward propagation output of the linear and ReLU activation functions will serve as a input here 
      Array size -> [No of samples x No of neurons required in the hidden and output layer]
      ------------------------------------------------------------------------
      Derivative:
      The derivative of the output w.r.t weights = input -> (1)
      The derivative of the output w.r.t biases = sum(1) -> (2) 
      The derivative of the output w.r.t input = weights -> (3) 
      ------------------------------------------------------------------------
      Output:
      weights_derv: (1) x derivatives + d(reg_loss w.r.t weights); Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases_derv: (2) x derivatives + d(reg_loss w.r.t biases); Array size -> [1 x No of neurons required in the hidden and output layer]
      inputs_derv: (3) x derivatives; Array size -> [No of samples x no of independent features]
      ========================================================================
      '''
      self.weights_derv = np.dot(self.inputs.T,derivatives)
      print('weight_derv',self.weights_derv)
      self.biases_derv = np.sum(derivatives, axis = 0, keepdims = True) #axis = 0 to sum columnwise, keepdim = True is to match the bias dimension
      print('biases_derv',self.biases_derv)
     
      # Regularization derivative at weights 
      if self.L2_weight_reg > 0:
         self.weights_derv += 2 * self.L2_weight_reg * self.weights
      
      # Regularization derivative at biases
      if self.L2_bias_reg > 0:
         self.biases_derv += 2 * self.L2_bias_reg * self.biases

      self.inputs_derv = np.dot(derivatives, self.weights.T)
      print('inputs_derv',self.inputs_derv)

      return self.weights_derv, self.biases_derv, self.inputs_derv
#
# ======================================================================
# Construction of Rectified Linear Units(ReLU) activation function
# ======================================================================
# 
class ReLU_Activation:
   '''
   ========================================================================
   Description:
   Performing forward and backward propagation of ReLU activation function
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
      return self.output

   def back_prop(self,derivatives):
      '''
      ========================================================================
      Description:
      Finding the derivative input of ReLU activation function.
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The backward propagation output of the dense layers will serve as a input here; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Derivative:
      The derivative of the ReLU activation function will be 1 for the entries in the input set that are greater than 0 and 0 if otherwise
      ------------------------------------------------------------------------
      Output:
      inputs_derv: d(ReLU activation function) x derivatives; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Note:
      -Size of the derivatives and inputs are the same 
      ========================================================================
      '''
      self.inputs_derv = derivatives.copy()
      self.inputs_derv[self.inputs <= 0] = 0
      return self.inputs_derv
#
# ======================================================================
# Construction of Linear activation function
# ======================================================================
#     
class Linear_Activation:
   '''
   ========================================================================
   Description:
   Performing forward and backward propagation of Linear activation function.
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

   def back_prop(self, derivatives):
      '''
      ========================================================================
      Description:
      Finding the derivative input of the linear activation function. 
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The backward propagation output of the loss functions will serve as a input here
      Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Derivative:
      The derivative of the linear activation function is 1
      ------------------------------------------------------------------------
      Output:
      inputs_derv: d(linear activation function) x derivatives = derivatives; Array size -> [No of samples x No of neurons required in the output layer]
      Like in forward propagation here to the input will be the output
      ========================================================================
      '''
      self.inputs_derv = derivatives.copy()
      return self.inputs_derv
#
# ======================================================================
# Construction of Mean squared error loss
# ======================================================================
# 
class MeanSquaredError_loss():
   '''
   ========================================================================
   Description:
   Performing forward and backward propagation of Mean squared error loss.
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
   
   def back_prop(self, derivatives, y):
      '''
      ========================================================================
      Description:
      Finding the derivative input of mean squared error loss.
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The forward propagation output of the linear activation functions
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
      reg_loss = 0 # Initially 0
      # Regularization at weights 
      if layer.L2_weight_reg > 0:
         reg_loss += layer.L2_weight_reg  * np.sum(layer.weights * layer.weights)
      # Regularization at biases 
      if layer.L2_bias_reg > 0:
         reg_loss += layer.L2_bias_reg  * np.sum(layer.biases * layer.biases)
      return reg_loss
#
# ======================================================================
# Construction of Root mean squared error loss
# ======================================================================
# 
class RootMeanSquaredError_loss():
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
      Computing loss using root mean squared error loss w.r.t predicted output and test metric.
      ------------------------------------------------------------------------
      Parameters:
      y_pred: The forward propagation output of the linear activation functions; Array size -> [No of samples x No of neurons required in the output layer]
      y: Test metric; Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Return:
      loss: loss value of thesample; dtype -> float
      ========================================================================
      '''
      lossPer_sample = np.sqrt(np.mean((y - y_pred)**2, axis = -1)) # Calculating the mean of each sample(if no of neurons > 1) and output it as an vector
      loss = np.mean(lossPer_sample)
      return loss
   
   def back_prop(self, derivatives, y):
      '''
      ========================================================================
      Description:
      Finding the derivative input of root mean squared error loss.
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The forward propagation output of the linear activation functions
      Array size -> [No of samples x No of neurons required in the output layer]
      y: Test metric(dependent feature y_train); Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Derivative:
      The derivative of the root mean squared error loss is taken w.r.t predicted output(y_pred)
      ------------------------------------------------------------------------
      Output:
      inputs_derv: d(root mean squared error loss) = (-1 * (test metric - derivatives) / sqrt((test metric - derivatives)**2) ) / no of samples 
      Array size -> [No of samples x No of neurons required in the output layer]
      ========================================================================
      '''
      n_samples =  1       # len(derivatives)    # Number of samples(rows)
      self.inputs_derv = (-1 * (y - derivatives)) / np.sqrt((y - derivatives)**2) 
      self.inputs_derv = self. inputs_derv / n_samples 
      return self.inputs_derv

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
      reg_loss = 0 # Initially 0
      # Regularization at weights 
      if layer.L2_weight_reg > 0:
         reg_loss += layer.L2_weight_reg  * np.sum(layer.weights * layer.weights)
      # Regularization at biases 
      if layer.L2_bias_reg > 0:
         reg_loss += layer.L2_bias_reg  * np.sum(layer.biases * layer.biases)
      return reg_loss
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

def back_prop_lossfunc(loss_function, activation_function, y , epsilon):
   '''
   ========================================================================
   Description:
   Performing backward propagation step for loss function and validating the results with gradient checking
   ------------------------------------------------------------------------
   Parameters:
   loss_function: Loss function to be used for back propagation
   activation_function: Predicted output
   y: Dependent feature
   epsilon: Small pertubation of 0.001; dtype -> float
   ------------------------------------------------------------------------
   Return:
   Assertion result 
   ========================================================================
   '''
   plusepsilon = loss_function.forward_prop(activation_function.output + epsilon,y)
   minusepsilon = loss_function.forward_prop(activation_function.output - epsilon,y)

   # Gradient check
   #------------------------------------------------------------------------
   GradCheck_BP_loss = gradient_checking(plusepsilon, minusepsilon,epsilon)

   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(loss_back_prop,GradCheck_BP_loss), 'test failed'

def back_prop_activationfunc(activation_function, derivative, Dense_layer, epsilon):
   '''
   ========================================================================
   Description:
   Performing backward propagation step for activation function and validating the results with gradient checking
   ------------------------------------------------------------------------
   Parameters:
   activation_function: The activation function to be used for back propagation
   derivative : Derivative input of chain rule
   Dense_layer: Forward propagation input of the activation function
   epsilon: Small pertubation of 0.001; dtype -> float
   ------------------------------------------------------------------------
   Return:
   Assertion result 
   ========================================================================
   '''
   actfun3_back_prop = activation_function.back_prop(derivative.inputs_derv) # Analytical solution

   plusepsilon_BP_actfun = activation_function.forward_prop(Dense_layer.output + epsilon)
   minusepsilon_BP_actfun = activation_function.forward_prop(Dense_layer.output - epsilon)

   # Gradient check with chain rule so multiplying with the input derivative
   #------------------------------------------------------------------------
   GradCheck_BP_actfun = gradient_checking(plusepsilon_BP_actfun, minusepsilon_BP_actfun,epsilon)*derivative.inputs_derv

   # Assertion
   #------------------------------------------------------------------------
   np.testing.assert_array_almost_equal(actfun3_back_prop,GradCheck_BP_actfun), 'test failed'

def back_prop_denselayers(Dense_layer,derivative,activation_function,epsilon):
   '''
   ========================================================================
   Description:
   Performing backward propagation step of dense layers and validating the results with gradient checking
   ------------------------------------------------------------------------
   Parameters:
   Dense_layer: The dense layer to be used for back propagation
   derivative : Derivative input of chain rule
   activation_function: Forward propagation input for the dense layer
   epsilon: Small perturbation of 0.001; dtype -> float
   ------------------------------------------------------------------------
   Return:
   Assertion results for weights, biases and inputs
   ========================================================================
   '''
   weightsBP, biasesBP, inputsBP = Dense_layer.back_prop(derivative.inputs_derv) # Analytical solution

   # Forward propagation step of dense layers added with pertubation and multiplied with input derivative because of chain rule
   #------------------------------------------------------------------------
   plusepsilon_weightsBP = (np.dot(activation_function.T,derivative.inputs_derv) * (Dense_layer.weights + epsilon)) + Dense_layer.biases 
   minuspsilon_weightsBP = (np.dot(activation_function.T,derivative.inputs_derv) * (Dense_layer.weights - epsilon)) + Dense_layer.biases
   plusepsilon_biasesBP = np.dot(activation_function ,Dense_layer.weights) + (Dense_layer.biases + epsilon) * derivative.inputs_derv
   minuspsilon_biasesBP = np.dot(activation_function ,Dense_layer.weights) + (Dense_layer.biases - epsilon) * derivative.inputs_derv
   plusepsilon_inputsBP = ((activation_function + epsilon) * np.dot(derivative.inputs_derv,Dense_layer.weights.T)) + Dense_layer.biases
   minuspsilon_inputsBP = ((activation_function - epsilon) * np.dot(derivative.inputs_derv,Dense_layer.weights.T)) + Dense_layer.biases

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

def read_parameters():
   '''
   ========================================================================
   Description:
   Reading in parameters such as input set, dependent feature and loss input for five test cases from the file: parameters.txt in the same directory 
   ------------------------------------------------------------------------
   Parameters:
   inputset: Input parameters; Array size -> [1 x 5]
   dependent feature: Dependent feature for each test case; dtype -> float
   loss: Loss input for which loss function to use for back propagation; dtype -> str
   ------------------------------------------------------------------------
   Return:
   param: Input set for pytest.mark.parametrize to check and varify all the test cases
   ------------------------------------------------------------------------
   Note:
   -The number of entries for the input set, dependent feature and loss are fixed and should not be modified
   ========================================================================
   '''
   input = []                                                     # Creating storage to store input set
   depFeat = []                                                   # Creating storage to store dependent feature
   loss_ip = []                                                   # Creating storage to store loss input

   # Setting path to open parameters file
   #------------------------------------------------------------------------
   cwd = os.path.basename(os.getcwd())                            # Locating current working directory
   if cwd == 'test_ANN_Back_prop':                                # If tested from the current directory
      filename = 'parameters.txt'
   elif cwd == 'test_ANN':                                        # If tested from test_ANN
      open_path = os.path.abspath('test_ANN_Back_prop')           
      filename = os.path.join(open_path,'parameters.txt')
   elif cwd == 'test_ML':                                         # If tested from test_ML
      open_path = os.path.abspath('test_ANN/test_ANN_Back_prop')           
      filename = os.path.join(open_path,'parameters.txt')
   else:                                                          # Else an error will be thrown stating that the user is in the wrong directory
      sys.exit('Error:Testing executed in the wrong directory\nPossible directories: test_ML, test_ANN and current directory of the file')
      
   with open(filename, 'r') as f:                                 # Read parameters
      read_param = f.readlines()
      for idx in np.arange(9,14,1):                               # Iterating over the fixed no of lines
         read_input = read_param[idx].split(' ')[0]               # Reading in the fixed no of input set
         read_expectedRes = read_param[idx].split(' ')[2]         # Reading in the fixed no of expected result
         read_lossip = read_param[idx].split(' ')[4]              # Reading in loss input to select loss function
         input.append(list(map(int,read_input.split(','))))       # Storing the input set
         depFeat.append(float(read_expectedRes))                  # Storing the dependent feature
         loss_ip.append(read_lossip.split('\n')[0])               # Storing loss input

   # Creating test cases for parametrize
   #------------------------------------------------------------------------
   param = [(np.array([input[0]]), depFeat[0], 0.001,loss_ip[0]),(np.array([input[1]]),depFeat[1], 0.001,loss_ip[1]),(np.array([input[2]]), depFeat[2], 0.001,loss_ip[2]),\
      (np.array([input[3]]),depFeat[3], 0.001,loss_ip[3]),(np.array([input[4]]), depFeat[4], 0.001,loss_ip[4])]
   return param

# Defining multiple test cases for testing
#------------------------------------------------------------------------
@pytest.mark.parametrize('input,y,epsilon,loss_input', read_parameters())

def test_CompleteBack_prop(input,y,epsilon,loss_input):
   '''
   ========================================================================
   Description:
   Test one complete backward propagation of the neural network by using numerical gradient
   ------------------------------------------------------------------------
   Parameters:
   input: Input parameters for each test case; Array size -> [1 x 5]
   y: Dependent feature for each test case; dtype -> float
   epsilon: Small pertubation of 0.001; dtype -> float
   loss_input: Loss input for which loss function to use for back propagation; dtype -> str
   ------------------------------------------------------------------------
   Note:
   -The number of neurons in both the hidden layers are fixed to three
   ========================================================================
   '''
   # Initialization  for no of layers, activation functions and loss function
   #------------------------------------------------------------------------
   dense_layer_1 =dense_layer(5,3)  
   activation_function_1 = ReLU_Activation()
   dense_layer_2 = dense_layer(3,3) 
   activation_function_2 = ReLU_Activation()
   dense_layer_3 = dense_layer(3,1) 
   activation_function_3 = Linear_Activation()

   # Select loss function
   #------------------------------------------------------------------------
   if loss_input == 'MSE':
      loss_function = MeanSquaredError_loss()
   elif loss_input == 'RMSE':
      loss_function = RootMeanSquaredError_loss()
   else:
      sys.exit('Error: Recheck loss input in file: parameters.txt in the directory: test_ANN_Back_prop')

   # Begin forward propagation
   #------------------------------------------------------------------------
   dense_layer_1.forward_prop(input) 
   activation_function_1.forward_prop(dense_layer_1.output) 
   dense_layer_2.forward_prop(activation_function_1.output) 
   activation_function_2.forward_prop(dense_layer_2.output)  
   dense_layer_3.forward_prop(activation_function_2.output)
   activation_function_3.forward_prop(dense_layer_3.output)

   # Test for first back propagation step : loss back prop
   back_prop_lossfunc(loss_function, activation_function_3, y , epsilon)

   # Test for second back propagation step : back prop of Activation function 3
   back_prop_activationfunc(activation_function_3, loss_function, dense_layer_3, epsilon)

   # Test for third back propagation step : back prop of Dense layer 3 
   back_prop_denselayers(dense_layer_3,activation_function_3,activation_function_2.output,epsilon)
   
   # Test for fouth back propagation step : back prop of Activation function 2 
   back_prop_activationfunc(activation_function_2, dense_layer_3, dense_layer_2, epsilon)
   
   # Test for fifth back propagation step : back prop of Dense layer 2 
   back_prop_denselayers(dense_layer_2,activation_function_2,activation_function_1.output,epsilon)
   
   # Test for sixth back propagation step : back prop of Activation function 1 
   back_prop_activationfunc(activation_function_1, dense_layer_2, dense_layer_1, epsilon)

   # Test for last back propagation step : back prop of Dense layer 1 
   back_prop_denselayers(dense_layer_1,activation_function_1,input,epsilon)
