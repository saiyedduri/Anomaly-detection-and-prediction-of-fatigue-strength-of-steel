'''
========================================================================
This program is a part of personal programming project
Library_no: 4
Library_name: PPP_ANN
------------------------------------------------------------------------
Machine learning algorithm: Artificial neural network(supervised machine learning algorithm for regression)
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
      dataset = pd.read_csv(data)      # Reading in the input dataset
      X = dataset.iloc[:,:-1]          # Selecting all the columns of the input dataset except the last(which is the dependent feature/ target feature)
      y = dataset.iloc[:,[-1]]         # Selecting the last column of the input dataset 
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

   def split_dataset(self,scaled_X,scaled_y):
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
      X_train.shape = (349,25) ----(80% of scaled_X)
      x_test.shape = (88,25)   ----(20% of scaled_X)
      ========================================================================
      '''
      X_train, X_test = np.split(scaled_X.sample(frac = 1,random_state = 0).values,[int(0.8*len(scaled_X))]) # Splitting the scaled independent features randomly 
      y_train, y_test = np.split(scaled_y.sample(frac = 1,random_state = 0).values,[int(0.8*len(scaled_y))]) # Splitting the scaled dependent feature randomly
      return X_train, X_test,y_train,y_test  
#
# ======================================================================
# Construction of dense layer
# ======================================================================
#   
class Dense_layer:
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
      This will be called upon in the main program when ever a layer is created
      ------------------------------------------------------------------------
      Parameters:
      inputs: No of independent features; dtype -> int
      n_neurons: No of neurons required in the hidden and output layer; dtype -> int
      L2_weight_reg: lamda hyperparameter @ weights; dtype -> float
      L2_bias_reg: lamda hyperparameter @ biases; dtype -> float
      ------------------------------------------------------------------------
      Outputs:
      weights: Random initialization of weights w.r.t Array of size -> [no of independent features x No of neurons required in the hidden and output layer]
      Biases: Set to zeros w.r.t Array of size -> [1 x No of neurons required in the hidden and output layer] 
      ------------------------------------------------------------------------
      Note:
      -Reducing the magnitude of the gaussian distribution by multiplying it with 0.01 
      -This will reduce the time taken by the model to fit the data during training
      -If the model is non-trainable biases should be assigned with some values 
      ========================================================================
      '''                           
      self.weights = 0.01 * np.random.randn(inputs,n_neurons)                # Creating random weights w.r.t the number of independent features and number of neurons
      self.biases = np.zeros((1,n_neurons))                                  # Creating biases w.r.t the number of neurons

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
      inputs: Input layer(independent features such as X_train or X_test) and activation function outputs; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Output:
      output: Hidden and output layer; Array size -> [No of samples of the input layer x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases     # Computing dot product between the inputs and weights and then adding the biases
     
   def back_prop(self,derivatives):
      '''
      ========================================================================
      Description:
      Finding the derivative inputs of weights,biases and inputs.
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The backward propagation output of the linear and ReLU activation functions will serve as a input here 
      Array size -> [No of samples of the input layer x No of neurons required in the hidden and output layer]
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
      ------------------------------------------------------------------------
      Note:
      -Size of the weights derivative will be equivalent to the size of the weights 
      -Size of the biases derivative will be equivalent to the size of the biases
      -Size of the inputs derivative will be equivalent to the size of the inputs
      ========================================================================
      '''
      self.weights_derv = np.dot(self.inputs.T, derivatives)            # Computing the derivative of the output w.r.t weights
      self.biases_derv = np.sum(derivatives, axis = 0, keepdims = True) # axis = 0 to sum columnwise, keepdim = True is to match the bias dimension

      # Regularization derivative at weights 
      if self.L2_weight_reg > 0:                                        # If only regularization weight hyperparameter is given as input perform the task
         self.weights_derv += 2 * self.L2_weight_reg * self.weights     # Add regularization weight to the computed weights derivative
      
      # Regularization derivative at biases
      if self.L2_bias_reg > 0:                                          # If only regularization bias hyperparameter is given as input perform the task
         self.biases_derv += 2 * self.L2_bias_reg * self.biases         # Add regularization bias to the computed biases derivative

      self.inputs_derv = np.dot(derivatives, self.weights.T)            # Computing the derivative of the output w.r.t input
      
   def read_weights_biases(self):
      '''
      ========================================================================
      Description:
      Reading in the updated weights and biases after training the model.
      ------------------------------------------------------------------------
      Return:
      weights: Updated weights of the model after training; Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases: Updated biases of the model after training; Array size -> [1 x No of neurons required in the hidden and output layer]
      ------------------------------------------------------------------------
      Note:
      -This reading in avoids re-training of the model to similar new datas and allows the model to act 
      as an pure predictor and perform predicitons 
      ========================================================================
      '''
      return self.weights,self.biases
#
# ======================================================================
# Construction of dense layer for pure predictor
# ======================================================================
# 
class Updated_layer:
   '''
   ========================================================================
   Description:
   Updated layer is same as the dense layer it is used when the model act as a pure predictor.
   ======================================================================== 
   '''

   def __init__(self,weights,biases):
      '''
      ========================================================================
      Description:
      Initializing fixed weights and biases. 
      ------------------------------------------------------------------------
      Parameters:
      weights: Reading in the Updated weights of a trained model; Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases: Reading in the Updated biases of a trained model; Array size -> [1 x No of neurons required in the hidden and output layer]
      ========================================================================
      '''                           
      self.weights = weights
      self.biases = biases

   def forward_prop(self,inputs):
      '''
      ========================================================================
      Description:
      Creating hidden layer with known inputs, fixed weights and biases.
      ------------------------------------------------------------------------
      Parameters:
      inputs: Input layer(scaled independent features) and activation function outputs; Array size -> [No of samples x no of independent features]
      ------------------------------------------------------------------------
      Output:
      output: Hidden layer; Array size -> [No of samples  x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases             # Computing dot product between the inputs and weights and then adding the biases
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
      inputs: Hidden layer; Array size -> [No of samples of the input layer x No of neurons required in the hidden layer]
      ------------------------------------------------------------------------
      Output:
      output: Negative values in the hidden layer are replaced by zeros; Array size -> [No of samples of the input layer x No of neurons required in the hidden layer]
      ------------------------------------------------------------------------
      Note:
      -ReLU activation function is used in the hidden layers and the size of the inputs and output are the same
      ========================================================================
      '''
      self.inputs = inputs
      self.output = np.maximum(0, inputs)    
   
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
      self.inputs_derv[self.inputs <= 0] = 0 # Assigning zeros to inputs_derv in the positions same as in inputs
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
      inputs: output layer; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Output:
      output: predicted output; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Note:
      -Linear activation function is used in the output layer
      ========================================================================
      '''
      self.inputs = inputs
      self.output = inputs

   def back_prop(self, derivatives):
      '''
      ========================================================================
      Description:
      Finding the derivative input of the linear activation function. 
      ------------------------------------------------------------------------
      Parameters:
      derivatives: The backward propagation output of the loss functions will serve as a input here
      Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Derivative:
      The derivative of the linear activation function is 1
      ------------------------------------------------------------------------
      Output:
      inputs_derv: d(linear activation function) x derivatives = derivatives; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      Like in forward propagation here to the input will be the output
      ========================================================================
      '''
      self.inputs_derv = derivatives.copy() 
#
# ======================================================================
# Construction of Loss calculation
# ======================================================================
# 
class LossCalculation:
   '''
   ========================================================================
   Description: Computing the loss and regularization loss of the model.
   ========================================================================
   '''

   def calculate_loss(self,output, y):
      '''
      ========================================================================
      Description:
      Computing the mean loss of loss per sample.
      ------------------------------------------------------------------------
      Parameters:
      output: The forward propagation output of the linear activation functions(also called as predicted output)
      Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      y: Test metric(dependent feature such as y_train and y_test); Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      -------------------------------------------------------------------------
      Return:
      loss: loss value of the model w.r.t predicted output and test metric; dtype -> float
      ========================================================================
      '''
      lossPer_sample = self.forward_prop(output, y)         # Computing loss per sample using MSE/ RMSE loss
      loss = np.mean(lossPer_sample)                        # Computing the mean loss for all the samples
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
      reg_loss = 0 # Initially 0

      # Regularization at weights 
      if layer.L2_weight_reg > 0:                                                  # If only regularization weight hyperparameter is given as input perform the task
         reg_loss += layer.L2_weight_reg  * np.sum(layer.weights * layer.weights)  # Computing regularization loss at weight
      
      # Regularization at biases 
      if layer.L2_bias_reg > 0:                                                    # If only regularization bias hyperparameter is given as input perform the task
         reg_loss += layer.L2_bias_reg  * np.sum(layer.biases * layer.biases)      # Computing regularization loss at bias

      return reg_loss
#
# ======================================================================
# Construction of Mean squared error loss
# ======================================================================
# 
class MeanSquaredError_loss(LossCalculation):
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
      Computing loss per sample using mean squared error loss w.r.t predicted output and test metric.
      ------------------------------------------------------------------------
      Parameters:
      y_pred: The forward propagation output of the linear activation functions; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
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
      derivatives: The forward propagation output of the linear activation functions
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
      n_samples = len(derivatives)                                 # Number of samples(rows)
      outputsPerSample = len(derivatives[0])                       # Number of neurons per sample(colums)
      self.inputs_derv = -2 * (y - derivatives) / outputsPerSample # Computing the derivative of MSE loss
      self.inputs_derv = self. inputs_derv / n_samples             # Normalizing the computed derivative by dividing it with the number of samples
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
      y_pred: The forward propagation output of the linear activation functions; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
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
      derivatives: The forward propagation output of the linear activation functions
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
      n_samples = len(derivatives)                      # Number of samples(rows)
      outputsPerSample = len(derivatives[0])            # Number of neurons per sample(colums)
      self.inputs_derv = (-1 * (y - derivatives) / outputsPerSample) / np.sqrt((y - derivatives)**2 / outputsPerSample) # Computing the derivative of RMSE loss
      self.inputs_derv = self. inputs_derv / n_samples  # Normalizing the computed derivative by dividing it with the number of samples
#
# ======================================================================
# Construction of Accuracy calculation
# ======================================================================
# 
class Accuracy:
   '''
   ========================================================================
   Description:
   Computing the accuracy of the model.
   ========================================================================
   '''  
   def coefficient_of_determination(self,y_pred,y):
      '''
      ========================================================================
      Description:
      Computing the coefficient of determination w.r.t predicted output and dependent feature.
      ------------------------------------------------------------------------
      Parameters:
      y_pred: The forward propagation output of the linear activation functions; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      y: Dependent feature such as y_train and y_test; Array size -> [No of samples of the input layer x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Return:
      Rsqr: It returns the proportion of variation between the dependent feature and the predicted output of the model; dtype -> float
      ------------------------------------------------------------------------
      Note :
      -The best result is when the predicted values exactly match the dependent feature i.e. R^2 = 1 (close enough)
      ========================================================================
      '''
      SSres = np.sum((y - y_pred)**2)          # Computing the residual sum of squares (1)
      SStot = np.sum((y- np.mean(y))**2)       # Computing the total sum of squares (2)
      Rsqr = 1 - (SSres/SStot)                 # Computing Rsqr w.r.t (1) and (2)
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
      Fixing a learning rate through out the training step is not an ideal solution as it may find local minimum
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
      layer: dense layers are given as inputs and are used to access the weights and biases of the layers
      ------------------------------------------------------------------------
      Output:
      weights: Updates the weights w.r.t the current learning rate and weights derivatives from backward propagation
      Array size -> [no of independent features x No of neurons required in the hidden and output layer]      
      biases: Updates the biases w.r.t the current learning rate and biases derivatives from backward propagation
      Array size -> [1 x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      layer.weights += -self.C_learning_R * layer.weights_derv   # Weights update for each layers with SGD optimizer
      layer.biases += -self.C_learning_R * layer.biases_derv     # Biases update for each layers with SGD optimizer
   
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
# Construction of Stochastic gradient descent with momentum Optimizer
# ======================================================================
# 
class SGD_Momentum_Optimizer:
   '''
   ========================================================================
   Description: Implementing Stochastic gradient descent with momentum Optimizer.
   ========================================================================
   '''
   
   def __init__(self, learning_R = 0.85, learning_R_decay = 0, momentum = 0):
      '''
      ========================================================================
      Parameters:
      learning_R: Hyperparameter learning rate; dtype -> float
      learning_R_decay: Hyperparameter learning rate decay; dtype -> float
      momentum: Hyperparameter momentum(momentum co-efficient or friction); dtype -> float
      ------------------------------------------------------------------------
      Note :
      -if learning_R_decay = 0 then the current learning rate = learning_R and there will be no update
      -if momentum = 0 then the SGDM = SGD
      ========================================================================
      '''
      self.learning_R = learning_R
      self.C_learning_R = learning_R                
      self.learning_R_decay = learning_R_decay
      self.itr = 0
      self.momentum = momentum
   
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
         self.C_learning_R = self.learning_R * (1. / (1. +self.learning_R_decay * self.itr))  # Computing current learning rate
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
      weights: Here weights update for each layer is done by subtracting the fraction of the previous parameter update retained using momentum with
      the current learning rate and weights derivatives from backward propagation --- (1)
      Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases: The same procedure is done in the biases update as in the weights update; Array size -> [1 x No of neurons required in the hidden and output layer] --- (2)
      ========================================================================
      '''
      if not hasattr(layer, 'weight_momentum'):               # Assigning momentum arrays by checking If there exist any momentum attribute in the dense layers
         layer.weight_momentum = np.zeros_like(layer.weights) # Initial value will be assigned as zeros for both weights and biases
         layer.bias_momentum = np.zeros_like(layer.biases)
      
      weight_increment = self.momentum * layer.weight_momentum - self.C_learning_R * layer.weights_derv  # ----- (1)
      layer.weight_momentum = weight_increment                                                           
      bias_increment = self.momentum * layer.bias_momentum - self.C_learning_R * layer.biases_derv       # ----- (2)
      layer.bias_momentum = bias_increment

      layer.weights += weight_increment                      # Weights update for each layers with SGDM optimizer 
      layer.biases += bias_increment                         # Biases update for each layers with SGDM optimizer
   
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
# Construction of Root mean square propagation Optimizer
# ======================================================================
# 
class RMSProp_Optimizer:
   '''
   ========================================================================
   Description: Implementing Root mean square propagation Optimizer.
   ========================================================================
   '''
   
   def __init__(self, learning_R = 1e-3, learning_R_decay = 0, epsilon = 1e-7, rho =0.9):
      '''
      ========================================================================
      Parameters:
      learning_R: Hyperparameter learning rate; dtype -> float
      learning_R_decay: Hyperparameter learning rate decay; dtype -> float
      epsilon: Hyperparameter for numerical stability; dtype -> float
      rho: Hyperparameter history memory decay rate; dtype -> float
      ------------------------------------------------------------------------
      Note :
      -if learning_R_decay = 0 then the current learning rate = learning_R and there will be no update
      ========================================================================
      '''
      self.learning_R = learning_R
      self.C_learning_R = learning_R                
      self.learning_R_decay = learning_R_decay
      self.itr = 0
      self.epsilon = epsilon
      self.rho = rho
   
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
         self.C_learning_R = self.learning_R * (1. / (1. + self.learning_R_decay * self.itr))  # Computing current learning rate
         return self.C_learning_R
   
   def parameters_update(self,layer):
      '''
      ========================================================================
      Description:
      Update the weights and biases after each epoch during training.
      ------------------------------------------------------------------------
      Parameters:
      layer: dense layers are given as inputs and are used to access the weights and biases of the layers
      ------------------------------------------------------------------------
      Output:
      weight_history: The update to the weight history is done by retaining a fraction of the previous history and adding it with the fraction of squared derivatives --- (1)
      weights: The weights update for each layer is done by dividing the the current learning rate and derivatives with sqrt of the weight_history added to epsilon --- (2)
      Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases: The same procedure is done in the biases update as in the weights update; Array size -> [1 x No of neurons required in the hidden and output layer] ---(3)
      ========================================================================
      '''
      if not hasattr(layer, 'weight_history'):               # Assigning history arrays by checking If there exist any history attribute in the dense layers
         layer.weight_history = np.zeros_like(layer.weights) # Initial value will be assigned as zeros for both weights and biases
         layer.bias_history = np.zeros_like(layer.biases)

      layer.weight_history = self.rho * layer.weight_history + (1 - self.rho) * layer.weights_derv**2  # --- (1)
      layer.bias_history = self.rho * layer.bias_history + (1 - self.rho) * layer.biases_derv**2       # --- (3)

      layer.weights += -self.C_learning_R * layer.weights_derv / (np.sqrt(layer.weight_history) + self.epsilon) # Epsilon - to avoid division by 0 --- (2)
      layer.biases += -self.C_learning_R * layer.biases_derv / (np.sqrt(layer.bias_history) + self.epsilon) # --- (3)
   
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
# Construction of Adaptive momentum Optimizer
# ======================================================================
# 
class Adam_Optimizer:
   '''
   ========================================================================
   Description: Implementing Adaptive momentum Optimizer.
   ========================================================================
   '''
   def __init__(self, learning_R = 1e-3, learning_R_decay = 0, epsilon = 1e-7, beta1 = 0.9, beta2 = 0.999):
      '''
      ========================================================================
      Parameters:
      learning_R: Hyperparameter learning rate; dtype -> float
      learning_R_decay: Hyperparameter learning rate decay; dtype -> float
      epsilon: Hyperparameter for numerical stability; dtype -> float
      beta1: Hyperparameter exponential decay rate for momentum mean; dtype -> float
      beta2: Hyperparameter exponential decay rate for momentum varience; dtype -> float
      ------------------------------------------------------------------------
      Note :
      -if learning_R_decay = 0 then the current learning rate = learning_R and there will be no update
      ========================================================================
      '''
      self.learning_R = learning_R
      self.C_learning_R = learning_R                # Current learning rate
      self.learning_R_decay = learning_R_decay
      self.itr = 0
      self.epsilon = epsilon
      self.beta1 = beta1
      self.beta2 = beta2

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
         self.C_learning_R = self.learning_R * (1. / (1. + self.learning_R_decay * self.itr)) # Computing current learning rate
         return self.C_learning_R

   def parameters_update(self,layer):
      '''
      ========================================================================
      Description:
      Update the weights and biases after each epoch during training.
      ------------------------------------------------------------------------
      Parameters:
      layer: dense layers are given as inputs and are used to access the weights and biases of the layers
      ------------------------------------------------------------------------
      Output:
      weight_momentum_M: The update to the weight momentum @ mean is done by retaining a fraction of the previous momentum and adding it with the fraction of derivatives -- (1)
      weight_momentum_V: The update to the weight momentum @ varience is done by retaining a fraction of the previous momentum and adding it with the fraction of squared derivatives --(2)
      C_weight_momentum_M & V: This is a bias correction mechanism that is applied in both mean and variance to compensate the zero values during the initial steps. This is 
      achieved by dividing both momentum mean and variance by 1- beta^epoch. As the epoch increases this beta^epoch will approach zero and will become divided by 1
      weights: The weights update for each layer is done by diving the the current learning rate and corrected momentum @ mean with sqrt of corrected momentum @ varience added to epsilon
      Array size -> [no of independent features x No of neurons required in the hidden and output layer]
      biases: The same procedure is done in the biases update as in the weights update; Array size -> [1 x No of neurons required in the hidden and output layer]
      ========================================================================
      '''
      if not hasattr(layer, 'weight_momentum_M'):                 # Assigning momentum arrays by checking If there exist any momentum attribute in the dense layers
         layer.weight_momentum_M = np.zeros_like(layer.weights)   # Initial value will be assigned as zeros for both weights and biases @ mean and varience
         layer.weight_momentum_V = np.zeros_like(layer.weights)  
         layer.bias_momentum_M = np.zeros_like(layer.biases) 
         layer.bias_momentum_V = np.zeros_like(layer.biases)
      
      layer.weight_momentum_M = self.beta1 * layer.weight_momentum_M + (1 - self.beta1) * layer.weights_derv # --- (1)
      layer.bias_momentum_M = self.beta1 * layer.bias_momentum_M + (1 - self.beta1) * layer.biases_derv
      
      C_weight_momentum_M = layer.weight_momentum_M / (1 - self.beta1 **(self.itr + 1)) # Corrected Weight momentum mean estimates for the initial zeroed values to speed up
      C_bias_momentum_M = layer.bias_momentum_M / (1 - self.beta1 **(self.itr + 1))     # the training in the initial epoches and eventually the corrected momentum_M = momentum_M

      layer.weight_momentum_V = self.beta2 * layer.weight_momentum_V + (1 - self.beta2) * layer.weights_derv**2 # --- (2)
      layer.bias_momentum_V = self.beta2 * layer.bias_momentum_V + (1 - self.beta2) * layer.biases_derv**2
      
      C_weight_momentum_V = layer.weight_momentum_V / (1 - self.beta2 **(self.itr + 1)) # Corrected Weight momentum varience estimates for the initial zeroed values to speed up
      C_bias_momentum_V = layer.bias_momentum_V / (1 - self.beta2 **(self.itr + 1))     # the training in the initial epoches and eventually the corrected momentum_V = momentum_V

      layer.weights += -self.C_learning_R * C_weight_momentum_M / (np.sqrt(C_weight_momentum_V) + self.epsilon) # Epsilon - to avoid division by 0
      layer.biases += -self.C_learning_R * C_bias_momentum_M / (np.sqrt(C_bias_momentum_V) + self.epsilon)
   
   def itr_update(self):
      '''
      ========================================================================
      Output:
      itr: Equivalent to epoch update during training; dtype -> int      
      ========================================================================
      '''
      self.itr += 1