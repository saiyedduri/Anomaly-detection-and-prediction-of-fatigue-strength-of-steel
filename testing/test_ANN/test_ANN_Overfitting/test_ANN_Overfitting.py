'''
========================================================================
This test program is a part of personal programming project -> test_ANN
Test_file: test_ANN_Overfitting.py
Test_type: Overfitting test
Aim: To check the correctness of implementation the model is tested for overfitting
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_ANN import Data_preprocessing
from PPP_ANN import Dense_layer
from PPP_ANN import ReLU_Activation,Linear_Activation
from PPP_ANN import SGD_Optimizer,SGD_Momentum_Optimizer,RMSProp_Optimizer,Adam_Optimizer
from PPP_ANN import MeanSquaredError_loss,RootMeanSquaredError_loss
from PPP_ANN import Accuracy

# Import required libraries
# -------------------------
import sys
import click
#
# ======================================================================
# Construction of the model ANN Regression
# ======================================================================
#
class ANN_Regression: 
   '''
   ========================================================================
   Description: 
   In this model class we compute coefficient of determination of the model after training and testing them.
   These results are then used to check whether the model is overfitting or not. 
   ========================================================================
   '''
   def __init__(self,data,no_of_epoch,hp,loss,optimizer):
      '''
      ========================================================================
      Description:
      Initializing the inputs and created libraries, performing data preprocessing steps and assigning three dense layers
      ------------------------------------------------------------------------
      Parameters:
      data: Input dataset in .csv format with dependent feature as the last column 
      no_of_epoch: Training size; dtype -> int
      hp: Hyperparameters as dictionary
      loss: Loss input; dtype -> str
      optimizer: Optimizer input; dtype -> str
      ------------------------------------------------------------------------
      Note:
      -The program by default runs with three dense layers
      ========================================================================
      '''
      self.no_of_epoch = no_of_epoch 
      self.hp = hp 
      self.loss_input = loss 
      self.optimizer_input = optimizer 

      # Assigning layers from hp dictionary
      self.hl1,self.hl2,self.opl = self.hp['layers']['hidden_layer1'],self.hp['layers']['hidden_layer2'],self.hp['layers']['output_layer']
     
      # Data_preprocessing
      # ------------------------------------------------------------------------
      Preprocessed_data = Data_preprocessing()      # Initialization for data preprocessing
      X,y = Preprocessed_data.import_dataset(data)  # X: Independent feature, y: dependent feature                 
      self.scaled_X,self.scaled_y,self.mean_y,self.std_y = Preprocessed_data.feature_scaling(X,y) # Scaled features, mean and std for rescaling during comparision
      self.X_train, self.X_test,self.y_train,self.y_test = Preprocessed_data.split_dataset(self.scaled_X,self.scaled_y) # Splitting dataset into training and test set 
   
      # Initialization of three dense layers and thus three activation functions. ReLU is used in the hidden layers 1 & 2 and linear in the output layer
      # Dense layer[1,2,3] inputs ->[No of independent features, hiddenlayer[1,2] / outputlayer[3] input, lamda hp input for regularization]
      #------------------------------------------------------------------------
      self.dense_layer_1 =Dense_layer(len(self.X_train[0]),self.hl1,self.hp['reg']['L2_weight_reg_hl1'],self.hp['reg']['L2_bias_reg_hl1']) 
      self.activation_function_1 = ReLU_Activation()
      self.dense_layer_2 = Dense_layer(len(self.dense_layer_1.weights[0]),self.hl2,self.hp['reg']['L2_weight_reg_hl2'],self.hp['reg']['L2_bias_reg_hl2'])
      self.activation_function_2 = ReLU_Activation()
      self.dense_layer_3 = Dense_layer(len(self.dense_layer_2.weights[0]),self.opl,self.hp['reg']['L2_weight_reg_hl3'],self.hp['reg']['L2_bias_reg_hl3']) 
      self.activation_function_3 = Linear_Activation()
      
      # Initialization for loss function and accuracy
      #------------------------------------------------------------------------
      self.loss_function_1 = MeanSquaredError_loss()
      self.loss_function_2 = RootMeanSquaredError_loss()
      self.model_accuracy = Accuracy()

      # Initialization for optimizers (Optimizer inputs are explained in PPP_ANN.py)
      #------------------------------------------------------------------------
      self.optimizer_1 = SGD_Optimizer(self.hp['SGD']['learning_R'], self.hp['SGD']['learning_R_decay'])
      self.optimizer_2 = SGD_Momentum_Optimizer(self.hp['SGDM']['learning_R'], self.hp['SGDM']['learning_R_decay'], self.hp['SGDM']['momentum'])
      self.optimizer_3 = RMSProp_Optimizer(self.hp['RMSP']['learning_R'], self.hp['RMSP']['learning_R_decay'], self.hp['RMSP']['epsilon'], self.hp['RMSP']['rho']) 
      self.optimizer_4 = Adam_Optimizer(self.hp['Adam']['learning_R'], self.hp['Adam']['learning_R_decay'], self.hp['Adam']['epsilon'], self.hp['Adam']['beta1'],\
      self.hp['Adam']['beta2'])                

   def calculate_loss_Function(self,loss_Function,dataset):
      '''
      ========================================================================
      Description:
      To calculate data loss, regularization loss and total loss.
      ------------------------------------------------------------------------
      Parameters:
      loss_Function: Loss_function_1 or loss_function_2. loss calculation are performed based on the loss function
      dataset: Dependent feature(y_train or y_test); Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Return:
      loss: It is the data loss w.r.t predicted output and dependent feature; dtype -> float
      reg_loss: Regularization loss computed from all three layers; dtype -> float
      total_loss: It is the sum of both loss and reg_loss; dtype -> float
      ========================================================================
      '''
      loss = loss_Function.calculate_loss(self.activation_function_3.output, dataset)
      reg_loss = loss_Function.regularization_loss(self.dense_layer_1) + loss_Function.regularization_loss(self.dense_layer_2)  + \
      loss_Function.regularization_loss(self.dense_layer_3)
      total_loss = loss + reg_loss
    
      return loss,reg_loss,total_loss
         
   def calculate_back_prop(self,loss_Function,dataset):
      '''
      ========================================================================
      Description:
      To perform backward propagation chain rule from loss function to dense layer 1.
      ------------------------------------------------------------------------
      Parameters:
      loss_Function: Loss_function_1 or loss_function_2. Different backward propagation procedure are fallowed w.r.t loss function input
      dataset: Dependent feature(y_train); Array size -> [No of samples x No of neurons required in the output layer]
      ------------------------------------------------------------------------
      Note:
      -The backward propagation starts from the loss function and goes backward through the pairs of activation functions and layers until the first layer
      -Backward propagation of dense layer creates derivatives of weights and biases which are used in optimizers
      -More details on back propagation and its inputs can be found in PPP_ANN.py
      ========================================================================
      '''
      loss_Function.back_prop(self.activation_function_3.output, dataset)  # Back prop of loss function(MSE or RMSE)
      self.activation_function_3.back_prop(loss_Function.inputs_derv)      # Back prop of linear activation function
      self.dense_layer_3.back_prop(self.activation_function_3.inputs_derv) # Back prop of dense layer 3
      self.activation_function_2.back_prop(self.dense_layer_3.inputs_derv) # Back prop of ReLU activation function
      self.dense_layer_2.back_prop(self.activation_function_2.inputs_derv) # Back prop of dense layer 2
      self.activation_function_1.back_prop(self.dense_layer_2.inputs_derv) # Back prop of ReLU activation function
      self.dense_layer_1.back_prop(self.activation_function_1.inputs_derv) # The final step is the back prop of dense layer 1

   def calculate_optimizer(self,optimizer):
      '''
      ========================================================================
      Description:
      To perform optimization and update parameters in the dense layers.
      ------------------------------------------------------------------------
      Parameters:
      optimizer: optimizer_1(SGD), optimizer_2(SGDM), optimizer_3(RMSP) or optimizer_4(Adam). Parameters update varies based on the optimizer input
      ------------------------------------------------------------------------
      Return:
      LearningR: Current learning rate from the optimizer; dtype -> float      
      ------------------------------------------------------------------------
      Note:
      -Derivatives of weights and biases which are used in optimizers to update weights and biases in the dense layer
      -More details on optimizer and its inputs can be found in PPP_ANN.py
      ========================================================================
      '''
      optimizer.learning_R_update()                      # Update learning rate in the optimizer
      LearningR= optimizer.C_learning_R                  # Return current learning rate from the optimizer
      optimizer.parameters_update(self.dense_layer_1)    # Update the parameter of dense layer 1 w.r.t optimizer
      optimizer.parameters_update(self.dense_layer_2)    # Update the parameter of dense layer 2 w.r.t optimizer
      optimizer.parameters_update(self.dense_layer_3)    # Update the parameter of dense layer 3 w.r.t optimizer
      optimizer.itr_update()                             # Update iteration step in the optimizer
      return LearningR
    
   def begin_training(self):
      '''
      ========================================================================
      Description:
      Training the model in a loop from forward propagation, backward propagation to optimization w.r.t the no of epoch.
      The loss and accuracy value of training results should be good enough as the model uses the update weights and biases to make predictions.
      ------------------------------------------------------------------------
      Note:
      -Parameters are updated in each dense layer after training
      ========================================================================
      '''
      # Select loss and optimizer from the input param dictionary
      #------------------------------------------------------------------------
      input_param = {'loss':{'MSE':self.loss_function_1, 'RMSE':self.loss_function_2},
                     'optimizer':{'SGD':self.optimizer_1, 'SGDM':self.optimizer_2, 'RMSP':self.optimizer_3, 'Adam':self.optimizer_4}}
      
      # Select loss function based on loss input
      #------------------------------------------------------------------------
      if self.loss_input in input_param['loss'].keys(): 
         loss_function =  input_param['loss'][self.loss_input]    

      # Select optimizer based on optimizer input
      #------------------------------------------------------------------------
      if self.optimizer_input in input_param['optimizer'].keys():
         optimizer =  input_param['optimizer'][self.optimizer_input] 

      for epoch in range(self.no_of_epoch): # Training begins w.r.t the no of epoch
        
         # Forward Propagation (More details on forward propagation and its inputs can be found in PPP_ANN.py)
         #------------------------------------------------------------------------
         self.dense_layer_1.forward_prop(self.X_train)                        # Forward prop of dense layer 1 with independent feature as an input
         self.activation_function_1.forward_prop(self.dense_layer_1.output)   # Forward prop of activation function 1 with dense layer 1 output as an input
         self.dense_layer_2.forward_prop(self.activation_function_1.output)   # Forward prop of dense layer 2 with activation function 1 output as an input
         self.activation_function_2.forward_prop(self.dense_layer_2.output)   # Forward prop of activation function 2 with dense layer 2 output as an input
         self.dense_layer_3.forward_prop(self.activation_function_2.output)   # Forward prop of dense layer 3 with activation function 2 output as an input
         self.activation_function_3.forward_prop(self.dense_layer_3.output)   # Forward prop of activation function 3 with dense layer 3 output as an input

         # Calculate Loss 
         #------------------------------------------------------------------------
         self.loss,reg_loss,total_loss = self.calculate_loss_Function(loss_function,self.y_train)
         
         # Calculate accuracy (More details on coefficient_of_determination and its inputs can be found in PPP_ANN.py)
         #------------------------------------------------------------------------
         self.Rsqr = self.model_accuracy.coefficient_of_determination(self.activation_function_3.output, self.y_train)
         
         # Back Propagation
         #------------------------------------------------------------------------ 
         self.calculate_back_prop(loss_function,self.y_train)

         # Perform Optimization
         #------------------------------------------------------------------------
         LearningR = self.calculate_optimizer(optimizer)
      
      # Forward prop for last optimizer update
      #-----------------------------------------------------------------------
      self.dense_layer_1.forward_prop(self.X_train)
      self.activation_function_1.forward_prop(self.dense_layer_1.output) 
      self.dense_layer_2.forward_prop(self.activation_function_1.output) 
      self.activation_function_2.forward_prop(self.dense_layer_2.output)  
      self.dense_layer_3.forward_prop(self.activation_function_2.output)
      self.activation_function_3.forward_prop(self.dense_layer_3.output)

      # Calculate Loss
      #----------------------------------------------------------------------- 
      self.loss,reg_loss,total_loss = self.calculate_loss_Function(loss_function,self.y_train)

      # Calculate accuracy
      #-----------------------------------------------------------------------
      self.Rsqr = self.model_accuracy.coefficient_of_determination(self.activation_function_3.output, self.y_train)

   def begin_prediction(self):
      '''
      ========================================================================
      Description:
      The updated parameters from training the model is now used to predict dependent feature from unseen independent features. 
      The predicted results are then tested against know dependent feature.
      ========================================================================
      '''
      # Select loss from the input param dictionary
      #-----------------------------------------------------------------------
      input_param = {'MSE':self.loss_function_1, 'RMSE':self.loss_function_2}

      # Select loss function based on loss input
      #------------------------------------------------------------------------
      if self.loss_input in input_param.keys():
         loss_function =  input_param[self.loss_input]
      
      self.pred_input = self.X_train         # To check for overfitting the trainined dataset X_train (independent features) will be the prediction input
      self.test_metric = self.y_train        # And y_train (dependent features) will be the testing metric
      
      # Forward Propagation (More details on forward propagation and its inputs can be found in PPP_ANN.py)
      #------------------------------------------------------------------------
      self.dense_layer_1.forward_prop(self.pred_input)                     # Forward prop of dense layer 1 with independent feature as an input
      self.activation_function_1.forward_prop(self.dense_layer_1.output)   # Forward prop of activation function 1 with dense layer 1 output as an input
      self.dense_layer_2.forward_prop(self.activation_function_1.output)   # Forward prop of dense layer 2 with activation function 1 output as an input
      self.activation_function_2.forward_prop(self.dense_layer_2.output)   # Forward prop of activation function 2 with dense layer 2 output as an input
      self.dense_layer_3.forward_prop(self.activation_function_2.output)   # Forward prop of dense layer 3 with activation function 2 output as an input
      self.activation_function_3.forward_prop(self.dense_layer_3.output)   # Forward prop of activation function 3 with dense layer 3 output as an input

      # Calculate Loss 
      #------------------------------------------------------------------------
      self.loss_pred,reg_loss,total_loss = self.calculate_loss_Function(loss_function,self.test_metric) # The loss will be calculated again with the test metric

      # Calculate accuracy (More details on coefficient_of_determination and its inputs can be found in PPP_ANN.py)
      #------------------------------------------------------------------------
      self.Rsqr_pred = self.model_accuracy.coefficient_of_determination(self.activation_function_3.output, self.test_metric) # The acc will be calculated again with the test metric
      
def check_inputs(no_of_epoch,loss,optimizer,layers):
   '''
   ========================================================================
   Description:
   To check whether certain user inputs are within the required limits.
   ------------------------------------------------------------------------
   Parameters:
   no_of_epoch: Training size; dtype -> int
   loss: Loss input; dtype -> str
   optimizer: Optimizer input; dtype -> str
   layer: Hiddenlayers; dtype -> int 
   ------------------------------------------------------------------------
   Note:
   -If the inputs are not within the options range, program exits by throwing an error with possible inputs
   ========================================================================
   '''
   # Checking whether the input is correct or wrong
   #------------------------------------------------------------------------
   if (not loss == 'MSE') and (not loss == 'RMSE'):                          # Checking whether the give loss input is within the options range
      sys.exit('Error: Recheck loss input\nPossible inputs: MSE or RMSE')    # Else program exits by throwing an error mentioning possible inputs
   if(no_of_epoch <= 0):                                # Checking epoch input, it should be greater than zero
      sys.exit('Error: no_of_epoch input must be > 0')  # Program exits if the input is lesser than or equal to zero
   if(layers[0]<=0 or layers[1]<=0):                    # Checking hidden layer inputs, it should be greater than zero
      sys.exit('Error: layers input must be > 0')       # Program exits if any one of the input is lesser than or equal to zero
   if(not optimizer == 'SGD') and (not optimizer == 'SGDM') and (not optimizer == 'RMSP') and (not optimizer == 'Adam'): # Checking whether the give optimizer input is within the options range
      sys.exit('Error: Recheck optimizer input\nPossible inputs: SGD,SGDM,RMSP or Adam') # Else program exits by throwing an error mentioning possible inputs
#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
def test_ANN_overfitting():
   @click.command()
   @click.option('--data',nargs=1,type=str,default='fatigue_dataset.csv',help='Enter input dataset.csv: last column must be the target feature')
   @click.option('--no_of_epoch',nargs=1,type=int,default=10001,help='Enter No of epoch for training(>0)')
   @click.option('--loss',nargs=1,type=str,default='MSE',help='Select loss: [MSE or RMSE]')
   @click.option('--optimizer',nargs=1,type=str,default='SGD',help='Select optimizer: [SGD,SGDM,RMSP or Adam]')
   @click.option('--layers',nargs=2,type=int,default=([10,10]),help='Enter hidden layer(N1,N2) for [IP-N1-N2-OP](>0)')
   @click.option('--reg',nargs=6,type=float,default=([2e-6,2e-6,3e-6,3e-6,0,0]),help='Enter regularization loss(l2) for the layers')
   @click.option('--sgd',nargs=2,type=float,default=([0.85,1e-1]),help='Enter SGD_optimizer input')
   @click.option('--sgdm',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input')
   @click.option('--rmsp',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input')
   @click.option('--adam',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input')
   #
   # ============================================================================================================================================
   #                                                      CREATING MODEL --> ANN REGRESSION
   # ============================================================================================================================================
   #
   def ANN_regression(data,no_of_epoch,loss,optimizer,layers,reg,sgd,sgdm,rmsp,adam):
      '''
      ========================================================================
      Description:
      This ANN REGRESSION model can be tested for eight different combinations of loss and optimizer input with one combination at a time.
      ========================================================================
      '''
      # Check certain user inputs
      #------------------------------------------------------------------------
      check_inputs(no_of_epoch,loss,optimizer,layers)

      # Adjustable hyperparameters must be tunned separately w.r.t the dataset used as it can impact the accuracy of the model
      # Creating the input dictionary of hp
      #------------------------------------------------------------------------     
      N1,N2 = layers                                  # Layer inputs in form of tuple
      L2_W1,L2_B1,L2_W2,L2_B2,L2_W3,L2_B3 = reg       # Regularization inputs in form of tuple
      SGD_lr,SGD_lrd = sgd                            # SGD hyperparameter inputs in form of tuple  @RMSE: SGD_lrd = 1e-2 so that the does not get stuck in local minima
      SGDM_lr,SGDM_lrd,SGDM_m = sgdm                  # SGDM hyperparameter inputs in form of tuple
      RMSP_lr,RMSP_lrd,RMSP_ep,RMSP_rho = rmsp        # RMSP hyperparameter inputs in form of tuple
      Adam_lr,Adam_lrd,Adam_ep,Adam_b1,Adam_b2 = adam # Adam hyperparameter inputs in form of tuple

      hyp_paramANN = {
            'layers':{'hidden_layer1':N1, 'hidden_layer2':N2, 'output_layer':1},
            'reg':{'L2_weight_reg_hl1':L2_W1, 'L2_bias_reg_hl1':L2_B1, 'L2_weight_reg_hl2':L2_W2, 'L2_bias_reg_hl2':L2_B2, 'L2_weight_reg_hl3':L2_W3, 'L2_bias_reg_hl3':L2_B3},
            'SGD':{'learning_R':SGD_lr, 'learning_R_decay':SGD_lrd},                                                          
            'SGDM':{'learning_R':SGDM_lr, 'learning_R_decay':SGDM_lrd, 'momentum':SGDM_m},                                             
            'RMSP':{'learning_R':RMSP_lr, 'learning_R_decay':RMSP_lrd, 'epsilon':RMSP_ep, 'rho':RMSP_rho},
            'Adam':{'learning_R':Adam_lr, 'learning_R_decay':Adam_lrd, 'epsilon':Adam_ep, 'beta1':Adam_b1, 'beta2':Adam_b2} } 

      # Initializing the model
      #------------------------------------------------------------------------ 
      model = ANN_Regression(data,no_of_epoch,hyp_paramANN,loss,optimizer)

      # Train the model
      #------------------------------------------------------------------------ 
      model.begin_training()

      # Test the model
      #------------------------------------------------------------------------                           
      model.begin_prediction()

      # Check for Overfitting
      #------------------------------------------------------------------------ 
      assert model.Rsqr == model.Rsqr_pred, 'test failed'
      print('AssertionPassed: R^2_training %f == R^2_testing %f'%(model.Rsqr,model.Rsqr_pred))
 

   if __name__ == '__main__':
      ANN_regression()

test_ANN_overfitting()