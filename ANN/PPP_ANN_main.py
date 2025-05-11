'''
========================================================================
This program is a part of personal programming project
ANN program No: 1 
file: PPP_ANN_main
------------------------------------------------------------------------
ANN algorithm is implemented to predict the dependent feature from the independent features
The program can train and test the model for a single combination of loss and optimizer
It can also be executed as an pure predictor for a single combination of loss and optimizer
To get a overall result of all the comibination execute PPP_ANN_createOverallresults.py 
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_ANN import Data_preprocessing
from PPP_ANN import Dense_layer,Updated_layer
from PPP_ANN import ReLU_Activation,Linear_Activation
from PPP_ANN import SGD_Optimizer,SGD_Momentum_Optimizer,RMSProp_Optimizer,Adam_Optimizer
from PPP_ANN import MeanSquaredError_loss,RootMeanSquaredError_loss
from PPP_ANN import Accuracy

# Import required libraries
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import click
import timeit
#
# ======================================================================
# Construction of the model ANN Regression
# ======================================================================
#
class ANN_Regression: 
   '''
   ========================================================================
   Description: 
   In this model class we train the model, plot training results, write trained parameters like weights and biases to a separate file, test the model, 
   write final results to a file to make combination plots later, plot testing results and finally perform prediction comparison between the predicted 
   results and the original results.
   ========================================================================
   '''

   def __init__(self,data,no_of_epoch,hp,loss,optimizer,writehl,dataset,predictor):
      '''
      ========================================================================
      Description:
      Initializing the inputs and created libraries, performing data preprocessing steps, Assigning three dense layers and 
      creating storage to store data.
      ------------------------------------------------------------------------
      Parameters:
      data: Input dataset in .csv format with dependent feature as the last column 
      no_of_epoch: Training size; dtype -> int
      hp: Hyperparameters as dictionary
      loss: Loss input; dtype -> str
      optimizer: Optimizer input; dtype -> str
      writehl: Input whether to write results into a separate file for different hidden layers; dtype -> int
      dataset: Input to select dataset among trainingset and testset for prediction; dtype -> str
      predictor: Predictor input whether to execute the model as a pure predictor; dtype -> str
      ------------------------------------------------------------------------
      Note:
      -To check for overfitting select trainingset in dataset input
      -The program by default runs with three dense layers(two hidden layer and an output layer)
      ========================================================================
      '''
      self.no_of_epoch = no_of_epoch 
      self.hp = hp 
      self.loss_input = loss 
      self.optimizer_input = optimizer 
      self.predictor = predictor 
      self.writehl = writehl  
      self.pred_dataset = dataset 

      # Assigning layers from hp dictionary
      self.hl1,self.hl2,self.opl = self.hp['layers']['hidden_layer1'],self.hp['layers']['hidden_layer2'],self.hp['layers']['output_layer']
     
      # Data_preprocessing
      # ------------------------------------------------------------------------
      Preprocessed_data = Data_preprocessing()      # Initialization for data preprocessing
      X,y = Preprocessed_data.import_dataset(data)  # X: Independent feature, y: dependent feature                 
      self.scaled_X,self.scaled_y,self.mean_y,self.std_y = Preprocessed_data.feature_scaling(X,y) # Scaled features, mean and std for rescaling during comparison
      self.X_train, self.X_test,self.y_train,self.y_test = Preprocessed_data.split_dataset(self.scaled_X,self.scaled_y) # Splitting dataset into training and test set 
   
      # Initialization of three dense layers and thus three activation functions. ReLU is used in the hidden layers 1 & 2 and linear in the output layer
      # Dense layer[1,2,3] inputs ->[No of independent features, hiddenlayer[1,2] / outputlayer[3] input, lamda hp input for regularization]
      #------------------------------------------------------------------------
      self.dense_layer_1 =Dense_layer(len(self.X_train[0]),self.hl1,self.hp['reg']['L2_weight_reg_hl1'],self.hp['reg']['L2_bias_reg_hl1']) 
      self.activation_function_1 = ReLU_Activation()   # Activation function at hidden layer 1
      self.dense_layer_2 = Dense_layer(len(self.dense_layer_1.weights[0]),self.hl2,self.hp['reg']['L2_weight_reg_hl2'],self.hp['reg']['L2_bias_reg_hl2'])
      self.activation_function_2 = ReLU_Activation()   # Activation function at hidden layer 2
      self.dense_layer_3 = Dense_layer(len(self.dense_layer_2.weights[0]),self.opl,self.hp['reg']['L2_weight_reg_hl3'],self.hp['reg']['L2_bias_reg_hl3']) 
      self.activation_function_3 = Linear_Activation() # Activation function at the output layer
      
      # Initialization for loss function and accuracy
      #------------------------------------------------------------------------
      self.loss_function_1 = MeanSquaredError_loss()       # Initializing MSE loss
      self.loss_function_2 = RootMeanSquaredError_loss()   # Initializing RMSE loss
      self.model_accuracy = Accuracy()                     # Initializing Coef. of determination

      # Initialization for optimizers SGD, SGDM, RMSP, Adam (Optimizer inputs are explained in PPP_ANN.py)
      #------------------------------------------------------------------------
      self.optimizer_1 = SGD_Optimizer(self.hp['SGD']['learning_R'], self.hp['SGD']['learning_R_decay']) # Initializing SGD Optimizer
      self.optimizer_2 = SGD_Momentum_Optimizer(self.hp['SGDM']['learning_R'], self.hp['SGDM']['learning_R_decay'], self.hp['SGDM']['momentum']) # Initializing SGDM Optimizer
      self.optimizer_3 = RMSProp_Optimizer(self.hp['RMSP']['learning_R'], self.hp['RMSP']['learning_R_decay'], self.hp['RMSP']['epsilon'], self.hp['RMSP']['rho']) # Initializing RMSP Optimizer
      self.optimizer_4 = Adam_Optimizer(self.hp['Adam']['learning_R'], self.hp['Adam']['learning_R_decay'], self.hp['Adam']['epsilon'], self.hp['Adam']['beta1'],\
      self.hp['Adam']['beta2']) # Initializing Adam Optimizer
      
      # Creating Storage to store results during training
      #------------------------------------------------------------------------
      self.stored_lossValues = np.zeros(self.no_of_epoch)     # To store data loss
      self.stored_reg_lossValues = np.zeros(self.no_of_epoch) # To store regularization loss
      self.stored_tot_lossValues = np.zeros(self.no_of_epoch) # To store the total loss i.e. data + regularization loss
      self.stored_LearningR = np.zeros(self.no_of_epoch)      # To store learning rate update
      self.stored_Coeff_R2 = np.zeros(self.no_of_epoch)       # To store accuracy 
      self.stored_trainingResult = []                         # To store predicted results after training
      self.stored_Parameters = []                             # To store parameters(weights and biases) after training                   

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
      ------------------------------------------------------------------------
      Note:
      -When the predictor is ON the program is executed as an pure predictor and only the loss is considered other losses are set to zero
      ========================================================================
      '''
      loss = loss_Function.calculate_loss(self.activation_function_3.output, dataset) # Calculating data loss
      if self.predictor == 'OFF': 
         reg_loss = loss_Function.regularization_loss(self.dense_layer_1) + loss_Function.regularization_loss(self.dense_layer_2)  + \
            loss_Function.regularization_loss(self.dense_layer_3)                     # Computing Regularization loss of all three layers
         total_loss = loss + reg_loss                                                 # Total loss = sum of both loss and reg_loss
      else:                                                                           # In case of pure predictor consider only data loss
         reg_loss,total_loss = 0,0
      return loss,reg_loss,total_loss
         
   def calculate_back_prop(self,loss_Function,dataset):
      '''
      ========================================================================
      Description:
      To perform backward propagation apply chain rule from loss function to dense layer 1.
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
      optimizer.parameters_update(self.dense_layer_1)    # Update the parameters of dense layer 1 w.r.t optimizer
      optimizer.parameters_update(self.dense_layer_2)    # Update the parameters of dense layer 2 w.r.t optimizer
      optimizer.parameters_update(self.dense_layer_3)    # Update the parameters of dense layer 3 w.r.t optimizer
      optimizer.itr_update()                             # Update iteration step in the optimizer
      return LearningR
    
   def begin_training(self):
      '''
      ========================================================================
      Description:
      Training the model in a loop from forward propagation, backward propagation to optimization w.r.t the no of epoch.
      The loss and accuracy value of training results should be good enough as the model uses the update weights and biases to make predictions.
      ------------------------------------------------------------------------
      Conditions:
      Make sure tabulate package is installed or pip install tabulate 
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Results_Training and writes a file: trainingresults_lossinput_optimizerinput.txt
      The file writes the complete history of training results such as accuracy, totalloss, loss, regularization loss and learning rate updates for each epoch 
      ------------------------------------------------------------------------
      Note:
      -Parameters are updated in each dense layer after training
      -Timetaken to perform forward prop, backward prop and optimizer is noted
      ========================================================================
      '''
      # Creating Storage to store time taken during forward prop, back prop and optimization 
      #------------------------------------------------------------------------
      self.timetaken_toforw = []  # Time taken during forward prop
      self.timetaken_toback = []  # Time taken during back prop
      self.timetake_byopt = []    # Time taken during optimization

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
         start_timeF = timeit.default_timer()                                 # Start timer
         self.dense_layer_1.forward_prop(self.X_train)                        # Forward prop of dense layer 1 with independent feature as an input
         self.activation_function_1.forward_prop(self.dense_layer_1.output)   # Forward prop of activation function 1 with dense layer 1 output as an input
         self.dense_layer_2.forward_prop(self.activation_function_1.output)   # Forward prop of dense layer 2 with activation function 1 output as an input
         self.activation_function_2.forward_prop(self.dense_layer_2.output)   # Forward prop of activation function 2 with dense layer 2 output as an input
         self.dense_layer_3.forward_prop(self.activation_function_2.output)   # Forward prop of dense layer 3 with activation function 2 output as an input
         self.activation_function_3.forward_prop(self.dense_layer_3.output)   # Forward prop of activation function 3 with dense layer 3 output as an input
         self.timetaken_toforw.append(timeit.default_timer()-start_timeF)     # Stop timer. The time taken to perform forward prop is noted

         # Calculate Loss 
         #------------------------------------------------------------------------
         self.loss,reg_loss,total_loss = self.calculate_loss_Function(loss_function,self.y_train)
         self.stored_lossValues[epoch] = self.loss       # Store data loss
         self.stored_reg_lossValues[epoch] = reg_loss    # Store regularization loss
         self.stored_tot_lossValues[epoch] = total_loss  # Store total loss
         
         # Calculate accuracy (More details on coefficient_of_determination and its inputs can be found in PPP_ANN.py)
         #------------------------------------------------------------------------
         self.Rsqr = self.model_accuracy.coefficient_of_determination(self.activation_function_3.output, self.y_train)
         self.stored_Coeff_R2[epoch] = self.Rsqr        # Store accuracy
         
         # Back Propagation
         #------------------------------------------------------------------------ 
         start_timeB = timeit.default_timer()                               # Start timer
         self.calculate_back_prop(loss_function,self.y_train)
         self.timetaken_toback.append(timeit.default_timer()-start_timeB)   # Stop timer. The time taken to perform back prop is noted

         # Perform Optimization
         #------------------------------------------------------------------------
         start_timeopt = timeit.default_timer()                             # Start timer
         LearningR = self.calculate_optimizer(optimizer)
         self.timetake_byopt.append(timeit.default_timer()-start_timeopt)   # Stop timer. The time taken to perform optimization is noted
         self.stored_LearningR[epoch]= LearningR                            # Store current learning rate

      # Forward prop for last optimizer update -> In the last epoch the optimizer gets updated and the loop exits 
      # Hence, generating result for forward prop with the updated parameters 
      #------------------------------------------------------------------------
      self.dense_layer_1.forward_prop(self.X_train)
      self.activation_function_1.forward_prop(self.dense_layer_1.output) 
      self.dense_layer_2.forward_prop(self.activation_function_1.output) 
      self.activation_function_2.forward_prop(self.dense_layer_2.output)  
      self.dense_layer_3.forward_prop(self.activation_function_2.output)
      self.activation_function_3.forward_prop(self.dense_layer_3.output)

      # Calculate Loss for last optimizer update
      #------------------------------------------------------------------------
      self.loss,reg_loss,total_loss = self.calculate_loss_Function(loss_function,self.y_train)
      self.stored_lossValues[epoch] = self.loss 
      self.stored_reg_lossValues[epoch] = reg_loss
      self.stored_tot_lossValues[epoch] = total_loss

      # Calculate accuracy for last optimizer update
      #------------------------------------------------------------------------
      self.Rsqr = self.model_accuracy.coefficient_of_determination(self.activation_function_3.output, self.y_train)
      self.stored_Coeff_R2[epoch] = self.Rsqr
      self.stored_LearningR[epoch] = optimizer.C_learning_R # Updating the current learning rate

      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Results_Training'):
         os.makedirs('Results_Training')
      save_path = os.path.abspath('Results_Training') # Save the file to the created directory

      # Creating a dataframe for the training results so that the results can be written in a tablulated form
      #------------------------------------------------------------------------
      writedata = pd.DataFrame({'Epoch': np.arange(0,self.no_of_epoch,1),'R^2':self.stored_Coeff_R2,'Totloss':self.stored_tot_lossValues,\
      'Loss': self.stored_lossValues,'Regloss': self.stored_reg_lossValues, 'lR': self.stored_LearningR})

      # writing training results for every 100th epoch into the file -> trainingresults_lossinput_optimizerinput.txt
      #------------------------------------------------------------------------
      with open(os.path.join(save_path,'trainingresults_%s_%s.txt' % (self.loss_input,self.optimizer_input)) , 'w') as f:
    
         print('Training results',file=f)                                                                      # Creating title infos before writing
         print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input)+'\n', file=f) # Details on loss and optimizer of the model
         print(writedata.iloc[::100,:].to_markdown(tablefmt='grid',index=False),file=f)                        # Tabulating the results in grid format without index
         print('\nAbbreviations:',file=f)                                                                      # Adding required abbreviations
         print('R^2: Coefficient of determination(Accuracy)\nTotLoss: total loss (Loss + Regloss)\nRegloss: regularization loss\nlR: learning rate',file=f)

      # store training results
      #------------------------------------------------------------------------
      self.stored_trainingResult.append(self.activation_function_3.output)
      
   def write_trainedparameters(self,writeparam_as):
      '''
      ========================================================================
      Description:
      To write the updated weights and biases after training which later used when the program runs as pure predictor.
      ------------------------------------------------------------------------
      Parameters:
      writeparam_as: selecting format to write from npy or txt; dtype -> str
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Results_Parameters and writes a file: parametersANN_lossinput_optimizerinput.npy / .txt
      The file writes updated weights and biases after training  
      ------------------------------------------------------------------------
      Note:
      -The file should be written in .npy format to read back in during prediciton
      -The format .txt is only for reading purpose
      ========================================================================
      '''
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Results_Parameters'):
            os.makedirs('Results_Parameters')
      save_path = os.path.abspath('Results_Parameters') # Save the file to the created directory

      # Store parameters
      #------------------------------------------------------------------------
      weights1,biases1 = self.dense_layer_1.read_weights_biases()                     
      self.stored_Parameters.append(weights1),self.stored_Parameters.append(biases1)  # Store updated weights and biases from dense layer 1
      weights2,biases2 = self.dense_layer_2.read_weights_biases()
      self.stored_Parameters.append(weights2),self.stored_Parameters.append(biases2)  # Store updated weights and biases from dense layer 2
      weights3,biases3 = self.dense_layer_3.read_weights_biases()
      self.stored_Parameters.append(weights3),self.stored_Parameters.append(biases3)  # Store updated weights and biases from dense layer 3

      # writing updated parameters into the file -> parametersANN_lossinput_optimizerinput.npy / .txt
      #------------------------------------------------------------------------   
      if writeparam_as == 'txt':
         with open(os.path.join(save_path,'parametersANN_%s_%s.txt'% (self.loss_input,self.optimizer_input)), 'w') as f:

            print('Parameters: Weights and biases (after training)',file=f)                                  # Creating title infos before writing
            print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input), file=f) # Details on loss and optimizer of the model

            for idx,dl in enumerate([1,1,2,2,3,3]): # Creating writing structure for reading in .txt format
               if not (idx)%2: 
                  print('\nDense layer %d: Weights\n'%(dl),file=f)
                  print(self.stored_Parameters[idx],file=f)
               else:
                  print('\nDense layer %d: Biases\n'%(dl),file=f)
                  print(self.stored_Parameters[idx],file=f)
      else:
         with open(os.path.join(save_path,'parametersANN_%s_%s.npy'% (self.loss_input,self.optimizer_input)), 'wb') as f:

            for idx in range(len(self.stored_Parameters)):
               np.save(f,self.stored_Parameters[idx]) # Saving the parameters in .npy format

   def write_dataforPlot(self):
      '''
      ========================================================================
      Description:
      To write certain results into a file in a well tabulated manner which are later used to make combination plots.
      ------------------------------------------------------------------------
      Conditions:
      Make sure tabulate package is installed or pip install tabulate
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Results_forPlotting and returns file1: plotdata_lossinput_optimizerinput.txt and file2: lossatHiddLayHL1HL2OL3_lossinput_optimizerinput.txt
      The file1 writes training results such as loss, learning rate update, accuracy and prediction result
      The file2 writes training loss and prediction result for different hiddenlayer inputs  
      ------------------------------------------------------------------------
      Note:
      -These written file are used in PPP_ANN_combination to make combination plots
      ========================================================================
      '''
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Results_forPlotting'):
            os.makedirs('Results_forPlotting')
      save_path = os.path.abspath('Results_forPlotting') # Save the file to the created directory

      # Creating a dataframe for the results to be ploted so that the results can be written in a tablulated form
      # .from_dict is used as the no of rows in all the results are not equal
      #------------------------------------------------------------------------
      writedata = pd.DataFrame.from_dict({'loss':self.stored_lossValues,'learning rate':self.stored_LearningR,'Coef of det':self.stored_Coeff_R2,\
                                'testing res':(self.activation_function_3.output).reshape(-1,)}, orient = 'index').T
      
      # writing results to plot into the file -> plotdata_lossinput_optimizerinput.txt
      #------------------------------------------------------------------------   
      with open(os.path.join(save_path,'plotdata_%s_%s.txt' % (self.loss_input,self.optimizer_input)) , 'w') as f:

         print('Plot data (after testing)',file=f)                                                              # Creating title infos before writing
         print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input)+'\n', file=f)  # Details on loss and optimizer of the model
         print(writedata.to_markdown(tablefmt='grid',index=False),file=f)                                       # Tabulating the results in grid format without index

      if self.writehl == 1: # Hidden layer results are written if only requested

         # Creating a dataframe for the results of different hiddenlayer inputs so that the results can be written in a tablulated form
         # .from_dict is used as the no of rows in all the results are not equal
         #------------------------------------------------------------------------
         writeHL = pd.DataFrame.from_dict({'loss':self.stored_lossValues,'testing res':(self.activation_function_3.output).reshape(-1,)}, orient = 'index').T

         # writing hiddenlayer results to plot into the file -> lossatHiddLayHL1HL2OL3_lossinput_optimizerinput.txt
         #------------------------------------------------------------------------   
         with open(os.path.join(save_path,'lossatHiddLay%d%d%d_%s_%s.txt' % (self.hl1,self.hl2,self.opl,self.loss_input,self.optimizer_input)) , 'w') as f:

            print('Plot data for Hidden layers in the architecture %d-%d-%d-%d  (after testing)'% (len(self.X_train[0]),self.hl1,self.hl2,self.opl),file=f) # Creating title infos before writing
            print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input)+'\n', file=f) # Details on loss and optimizer of the model
            print(writeHL.to_markdown(tablefmt='grid',index=False),file=f)                                        # Tabulating the results in grid format without index

   def read_parameters(self):
      '''
      ========================================================================
      Description:
      If the model is running as a pure predictor the model reads in the writen updated weights and biases from the
      file: parametersANN_lossinput_optimizerinput.npy in the directory: Results_Parameters.
      ------------------------------------------------------------------------
      Conditions:
      The dataset used when the model runs as a predictor should be similar to or an extension of the datas that are used for training the model
      ------------------------------------------------------------------------
      Return:
      parameters: return the read in updated weights and biases of three dense layers after training as a list
      ------------------------------------------------------------------------
      Note / Example :
      -When the model runs as a predictor i.e. when predictor is ON the model doesnot perform training
      ========================================================================
      '''
      open_path = os.path.abspath('Results_Parameters')                                                   # Specify the directory to open the file
      filename = os.path.join(open_path,'parametersANN_%s_%s.npy'%(self.loss_input,self.optimizer_input)) # Accessing the files w.r.t the combinations(loss and optimizer)
      with open(filename, 'rb') as f:                              # Read in .npy format file
         weight1,bias1 = np.load(f), np.load(f)                    # Loading in the updated parameters of dense layer 1
         weight2,bias2 = np.load(f), np.load(f)                    # Loading in the updated parameters of dense layer 2
         weight3,bias3 = np.load(f), np.load(f)                    # Loading in the updated parameters of dense layer 3
      parameters = [weight1,bias1,weight2,bias2,weight3,bias3]     # returning the parameters in form of a list 
      return parameters

   def begin_prediction(self):
      '''
      ========================================================================
      Description:
      The updated parameters from training the model is now used to predict dependent feature from unseen independent features. 
      The predicted results are then tested against know dependent feature.
      ------------------------------------------------------------------------
      Conditions:
      The model should be executed as a predictor only after training the model with dataset similar to that of the predictor dataset
      The layers,loss and optimizer used during predictor should be same as the ones used while training the model
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Results_Training or Results_ANN_predictor depending on predictor input and writes a file: testingresults_lossinput_optimizerinput.txt
      The file writes the testing results such as accuracy and loss of the model
      ------------------------------------------------------------------------
      Note:
      -The idea behind the predictor is to avoid re-training of the model if prediction has to be done on new datasets that are similar to or an extension 
      of the datasets that was used during the model training  
      -So when the model runs as a predictor no training will be done and will consider the input dataset directly for prediction
      ========================================================================
      '''
      # Select loss from the input param dictionary
      #-----------------------------------------------------------------------
      input_param = {'MSE':self.loss_function_1, 'RMSE':self.loss_function_2}

      # Select loss function based on loss input
      #------------------------------------------------------------------------
      if self.loss_input in input_param.keys():
         loss_function =  input_param[self.loss_input]
      
      if self.predictor == 'OFF':                  # Program runs normally when the predictor is OFF
         # Select dataset to perform prediction
         #------------------------------------------------------------------------
         if self.pred_dataset == 'testset':        # To test the performance of the trained parameters
            self.pred_input = self.X_test          # For testset X_test (independent features) will be the prediction input
            self.test_metric = self.y_test         # And y_test (dependent features) will be the testing metric or original output
         elif self.pred_dataset == 'trainingset':  # To test whether the model overfits(model correctness: The model should overfit)
            self.pred_input = self.X_train         # In that case the trainined dataset X_train (independent features) will be the prediction input
            self.test_metric = self.y_train        # And y_train (dependent features) will be the testing metric or original output
         
      elif self.predictor == 'ON':                                          # When the predictor is ON we use the read in parameters to perform prediction 
         parameters = self.read_parameters()                                        
         self.dense_layer_1 = Updated_layer(parameters[0],parameters[1])    # Recreating dense layer 1 by using the parameters of dense layer 1 in updated layer
         self.dense_layer_2 = Updated_layer(parameters[2],parameters[3])    # Recreating dense layer 2 by using the parameters of dense layer 2 in updated layer
         self.dense_layer_3 = Updated_layer(parameters[4],parameters[5])    # Recreating dense layer 3 by using the parameters of dense layer 3 in updated layer
         self.pred_input = self.scaled_X.values                             # Considering scaled independent features of the predictor dataset as predictor input
         self.test_metric = self.scaled_y.values                            # Considering scaled dependent feature of the predictor dataset as testing metric
      
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
      
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      path = 'Results_ANN_predictor' if self.predictor == 'ON' else 'Results_Testing' # If runs as a pure predictor save testing results to -> Results_ANN_predictor
      if not os.path.exists(path):
            os.makedirs(path)
      save_path = os.path.abspath(path) # Save the file to the created directory

      # writing testing results into the file -> testingresults_lossinput_optimizerinput.txt
      #------------------------------------------------------------------------
      with open(os.path.join(save_path,'testingresults_%s_%s.txt' % (self.loss_input,self.optimizer_input)) , 'w') as f:

         print('Testing results',file=f)                                                                       # Creating title infos
         print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input)+'\n', file=f) # Details on loss and optimizer of the model
         print('testing:  ' + f'R^2: {self.Rsqr_pred}, ' + f'loss: {self.loss_pred}',file=f)                   # Writing prediction accuracy and loss
         print('\nAbbreviations:',file=f)                                                                      # Adding required abbreviations
         print('R^2: Coefficient of determination(Accuracy)',file=f)
      
   def prediction_comparison(self):
      '''
      ========================================================================
      Description:
      To write result comparison between the predicted and the target result in a well tabulated manner.
      ------------------------------------------------------------------------
      Conditions:
      Make sure tabulate package is installed or pip install tabulate
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Results_TargetVSpred or Results_ANN_predictor depending on predictor input and writes a file: resultcomparision_lossinput_optimizerinput.txt 
      The file writes the testing results and the test metric of the model
      ========================================================================
      '''
      # Rescaling computed prediction and target values
      #------------------------------------------------------------------------
      rescaled_y_test = (self.test_metric*self.std_y + self.mean_y).reshape(-1,)          # Using the mean and std from scaling to rescale the target result as per dataset
      rescaled_y_pred = (self.activation_function_3.output*self.std_y + self.mean_y).reshape(-1,) # Using the mean and std from scaling to rescale the predicted result as per dataset
      
      # To check the correctness of rescaling, accuracy is computed for the rescaled results and then compared with the original accuracy result
      #------------------------------------------------------------------------
      rescaled_Rsqr = self.model_accuracy.coefficient_of_determination(rescaled_y_pred, rescaled_y_test)
      
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      path = 'Results_ANN_predictor' if self.predictor == 'ON' else 'Results_TargetVSpred' # If runs as a pure predictor save comparison results to -> Results_ANN_predictor
      if not os.path.exists(path):
            os.makedirs(path)
      save_path = os.path.abspath(path) # Save the file to the created directory
      
      # Creating a dataframe for comparison results so that the results can be written in a tablulated form
      #------------------------------------------------------------------------
      writedata = pd.DataFrame({'target':rescaled_y_test,'predicted':rescaled_y_pred})

      # writing comparison of the results and the accuracy into the file -> resultcomparision_lossinput_optimizerinput.txt
      #------------------------------------------------------------------------ 
      with open(os.path.join(save_path, 'resultcomparison_%s_%s.txt' % (self.loss_input,self.optimizer_input)), 'w') as f:

         print('Result comparison: Target vs Predicted',file=f)                                                   # Creating title infos before writing
         print('\nLoss and optimizer used: '+str(self.loss_input)+', '+str(self.optimizer_input)+'\n', file=f)    # Details on loss and optimizer of the model 
         print(writedata.to_markdown(tablefmt='grid',floatfmt='.0f',index=False),file=f)                          # Tabulating the results in grid format without index

         print('\nComparing R^2 for correctness: ' + f'prediction: {self.Rsqr_pred}, vs ' + f're-scaled prediction: {rescaled_Rsqr}',file=f) # checking for rescaling correctness
         print('\nAbbreviations:',file=f)                                                                         # Adding required abbreviations
         print('R^2: Coefficient of determination(Accuracy)',file=f)
         if self.predictor == 'OFF':                                                                              # Program runs normally when the predictor is OFF
            print('\nTime taken in seconds:\nForward propagation = %f; Backward propagation = %f; Optimization = %f'%(np.mean(self.timetaken_toforw),\
               np.mean(self.timetaken_toback),np.mean(self.timetake_byopt)), file=f)           # Writing time taken results of forward prop, back prop and optimizer
   
   def write_time(self,timetaken_permodel,timetaken_totrain,timetaken_topred):
      '''
      ========================================================================
      Description:
      Writing time taken results for training, prediction and entire model to execute for each combinations
      ------------------------------------------------------------------------
      Output:
      Creates a directory if one does not exist: Results_TargetVSpred or Results_ANN_predictor depending on predictor input 
      and append time taken results to the file: resultcomparision_lossinput_optimizerinput.txt
      ------------------------------------------------------------------------
      Note:
      If the predictor is ON the time taken for the model to train is ignored 
      ========================================================================
      '''
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      path = 'Results_ANN_predictor' if self.predictor == 'ON' else 'Results_TargetVSpred' # If runs as a pure predictor append time taken results to -> Results_ANN_predictor
      if not os.path.exists(path):
         os.makedirs(path)
      save_path = os.path.abspath(path) # Save the file to the created directory

      # Write timing results for each combination -> Model, training and prediction
      #------------------------------------------------------------------------
      with open(os.path.join(save_path,'resultcomparison_%s_%s.txt' % (self.loss_input,self.optimizer_input)) , 'a') as f:

         if self.predictor == 'OFF':
            print('Model = %f; Training = %f; Prediction = %f' %(timetaken_permodel,timetaken_totrain,timetaken_topred), file=f)
         else:
            print('\nTime taken in seconds:\nModel = %f; Prediction = %f' %(timetaken_permodel,timetaken_topred), file=f)

   def plot_trainingresults(self):
      '''
      ========================================================================
      Description:
      Creating a plot for training results which includes training ouput,target result for training, loss, accuracy and learning rate update.
      Each plot depends on the loss and optimizer input.
      ------------------------------------------------------------------------
      Output:
      Creates a directory: Plots_ANN_Main and saves a plot: plot_trainingresults_lossinput_optimizerinput.png
      ========================================================================
      '''
      # Reassigning loss and optimizer for convenience 
      self.ls = self.loss_input
      self.op = self.optimizer_input

      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Plots_ANN_Main'):
            os.makedirs('Plots_ANN_Main')
      save_path = os.path.abspath('Plots_ANN_Main') # Save the file to the created directory
     
      # Plot for Training set results
      #------------------------------------------------------------------------
      plt.suptitle('Training result (@'+str(self.ls)+ ' & '+str(self.op)+')',fontsize = 24) # Creating suptitle for the plot
      plt.subplots_adjust(top=0.90)                                                         # Adjusting the title position
      ax1 = plt.subplot2grid((2,3),(0,0), colspan = 3)                                      # Plotting in grids -> occupy all the three columns of the first row
      ax1.set_title('Target vs Trained')                                                    # Set plot title
      ax1.set_xlabel('No of samples')                                                       # Set label for x-axis -> will be the no of sample 
      ax1.set_ylabel('Sample values')                                                       # Set label for y-axis -> will be the values of each sample
      ax1.plot(self.y_train*self.std_y + self.mean_y,'o--',markersize = 7,color = 'red',label = 'Target data') # plotting training and target results
      ax1.plot(self.stored_trainingResult[0]*self.std_y + self.mean_y,'o-',markersize = 4,color = 'blue',label='Trained data ($R^2$ = %.3f; loss = %.4f)'%(self.Rsqr,self.loss))
      ax1.legend()                                                                          

      # Creating label for loss plot w.r.t the loss and optimizer input
      #------------------------------------------------------------------------
      if self.op == 'SGD':
         self.ex_label = 'loss = %.4f'%(self.loss)
      elif self.op == 'SGDM':
         self.ex_label = 'loss = %.4f; m = '%(self.loss)+str(self.hp[self.op]['momentum']) 
      elif self.op == 'RMSP':
         self.ex_label = 'loss = %.4f; \u03B5 = '%(self.loss)+str(self.hp[self.op]['epsilon'])+'; \u03C1 = '+str(self.hp[self.op]['rho'])                   
      elif self.op == 'Adam':
         self.ex_label = 'loss = %.4f; \u03B5 = '%(self.loss)+str(self.hp[self.op]['epsilon'])+'; \u03B21 = '+str(self.hp[self.op]['beta1'])+'; \u03B22 = '\
         +str(self.hp[self.op]['beta2'])

      # Plot for training loss
      #------------------------------------------------------------------------
      ax2 = plt.subplot2grid((2,3),(1,0))                                                 # Plotting in grids -> last row first column
      ax2.set_title('Loss' )                                                              # Set plot title
      ax2.set_xlabel('No of epoch')                                                       # Set label for x-axis -> will be the no of epoch
      ax2.set_ylabel('Loss value')                                                        # Set label for y-axis -> will be loss values of training
      ax2.plot(np.arange(0,self.no_of_epoch,1),self.stored_lossValues, label = self.ex_label)
      ax2.legend()
      
      # Plot for Coefficient of determination during training
      #------------------------------------------------------------------------
      ax3 = plt.subplot2grid((2,3),(1,1))                                                 # Plotting in grids -> last row second column
      ax3.set_title('Coefficient of determination')                                       # Set plot title
      ax3.set_xlabel('No of epoch')                                                       # Set label for x-axis -> will be the no of epoch
      ax3.set_ylabel('$R^2$ value')                                                       # Set label for y-axis -> will be accuracy values of training
      ax3.plot(np.arange(0,self.no_of_epoch,1),self.stored_Coeff_R2,label = '$R^2$ = %.3f'% self.Rsqr)
      ax3.legend()

      # Create label for learning rate w.r.t optimizer input
      #------------------------------------------------------------------------
      ex_label = 'lr = '+str(self.hp[self.op]['learning_R'])+'; lr_d = '+str(self.hp[self.op]['learning_R_decay'])
      
      # Plot for Learning rate update during training
      #------------------------------------------------------------------------
      ax4 = plt.subplot2grid((2,3),(1,2))                                                  # Plotting in grids -> last row last column
      ax4.set_title('Learning rate')                                                       # Set plot title
      ax4.set_xlabel('No of epoch')                                                        # Set label for x-axis -> will be the no of epoch
      ax4.set_ylabel('Learning rate value')                                                # Set label for y-axis -> will be learning rate values of training
      ax4.plot(np.arange(0,self.no_of_epoch,1),self.stored_LearningR,label = ex_label)
      ax4.legend()
      fig = plt.gcf()                                                                      # Get the current figure
      fig.set_size_inches((22.5 ,11.5),forward = False)                                    # Assigning plot size (width,height)
      fig.set_dpi(300)                                                                     # Set resolution in dots per inch
      fig.savefig(os.path.join(save_path,'plot_trainingresults_%s_%s.png'%(self.loss_input,self.optimizer_input)),bbox_inches='tight') # Saving the plot
      plt.clf()                                                                            # Cleaning the current figure after saving

   def plot_testingresults(self):
      '''
      ========================================================================
      Description:
      Creating a plot for testing results which includes predicted and the target result.
      Each plot depends on the loss and optimizer input.
      ------------------------------------------------------------------------
      Output:
      Creates a directory if one does not exists: Plots_ANN_Main and returns a plot: plot_testingresults_lossinput_optimizerinput.png    
      ========================================================================
      '''
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Plots_ANN_Main'):
            os.makedirs('Plots_ANN_Main')
      save_path = os.path.abspath('Plots_ANN_Main') # Save the file to the created directory

      # Plot for Testing set results
      #------------------------------------------------------------------------
      plt.title('Testing result: Target vs Prediction(@'+str(self.loss_input)+ ' & '+str(self.optimizer_input)+')') # Set plot title
      plt.xlabel('No of samples')                                                                                   # Set label for x-axis -> will be the no of sample
      plt.ylabel('Sample values')                                                                                   # Set label for y-axis -> will be the values of each sample
      plt.plot(self.test_metric*self.std_y + self.mean_y,'o--',markersize = 7,color = 'red',label = 'Target data')  # Plotting testing and target results
      plt.plot(self.activation_function_3.output*self.std_y + self.mean_y,'o-',markersize = 4,color='blue',label='Predicted data ($R^2$ = %.3f; loss = %.4f)'%(self.Rsqr_pred,self.loss_pred))
      plt.legend(loc='lower left')                                                                                  
      fig = plt.gcf()                                                                                                # Get the current figure
      fig.set_size_inches((9.5,7),forward = False)                                                                   # Assigning plot size (width,height) 
      fig.set_dpi(300)                                                                                               # Set resolution in dots per inch
      fig.savefig(os.path.join(save_path,'plot_testingresults_%s_%s.png'%(self.loss_input,self.optimizer_input)),bbox_inches='tight') # Saving the plot
      plt.clf()                                                                                                      # Cleaning the current figure after saving

   def plot_testingerror(self):
      '''
      ========================================================================
      Description:
      Creating a error plot for the error between the target and the predicted results.
      Each plot depends on the loss and optimizer input.
      ------------------------------------------------------------------------
      Output:
      Creates a directory if one does not exists: Plots_ANN_Main and returns a plot: plot_testingerror_lossinput_optimizerinput.png     
      ------------------------------------------------------------------------
      ========================================================================
      '''
      # Creating a directory if one does not exists
      #------------------------------------------------------------------------
      if not os.path.exists('Plots_ANN_Main'):
               os.makedirs('Plots_ANN_Main')
      save_path = os.path.abspath('Plots_ANN_Main') # Save the file to the created directory
      
      # Plot for Testing set error
      #------------------------------------------------------------------------
      plt.title('Testing set error(@'+str(self.loss_input)+ ' & '+str(self.optimizer_input)+')') # Set plot title
      plt.xlabel('No of samples')                                                                # Set label for x-axis -> will be the no of sample
      plt.ylabel('Error value')                                                                  # Set label for y-axis -> error value
      Error = (self.test_metric*self.std_y + self.mean_y)-(self.activation_function_3.output*self.std_y + self.mean_y) # Compute error
      plt.plot(Error, 'o--',markersize = 7,color = 'red', label = 'Error')                       # Plotting testing set error
      plt.axhline(y=0, color = 'blue', linestyle = '--')                                         # Make horizontal line at zero error
      plt.legend()
      fig = plt.gcf()                                                                            # Get the current figure
      fig.set_size_inches((9.5 ,7),forward = False)                                              # Assigning plot size (width,height)
      fig.set_dpi(300)                                                                           # Set resolution in dots per inch
      fig.savefig(os.path.join(save_path,'plot_testingerror_%s_%s.png'%(self.loss_input,self.optimizer_input)),bbox_inches='tight') # Saving the plot
      plt.clf()                                                                                  # Cleaning the current figure after saving


def check_inputs(no_of_epoch,pred_dataset,loss,optimizer,layers,makeplot,writeparam_as,writehl,predictor):
   '''
   ========================================================================
   Description:
   To check whether certain user inputs are within the required limits.
   ------------------------------------------------------------------------
   Parameters:
   no_of_epoch: Training size; dtype -> int
   pred_dataset: Input to select dataset among trainingset and testset for prediction; dtype -> str
   loss: Loss input; dtype -> str
   optimizer: Optimizer input; dtype -> str
   layer: Hiddenlayers; dtype -> int 
   makeplot: Input whether to make plots or not; dtype -> int
   writeparam_as: Writing format input to write updated parameters; dtype -> str 
   writehl: Input whether to write results into a separate file for different hidden layers; dtype -> int
   predictor: Predictor input whether to execute the model as an pure predictor; dtype -> str
   ------------------------------------------------------------------------
   Note:
   -If the inputs are not within the options range, program exits by throwing an error with possible inputs
   ========================================================================
   '''
   # Checking whether the input is correct or wrong
   #------------------------------------------------------------------------
   inputset = [pred_dataset,loss,makeplot,writeparam_as,writehl,predictor] # Grouping similar inputs and their options togather
   optionset = [['testset','trainingset','pred_dataset'],['MSE','RMSE','loss'],[0,1,'makeplot'],['npy','txt','writeparam_as'],[0,1,'writehl'],['ON','OFF','predictor']]

   for idx,input in enumerate(inputset): # Checking for correctness
      if (not input == optionset[idx][0]) and (not input == optionset[idx][1]): # If the inputs are not within the options range program exits by throwing an error mentioning possible inputs
         sys.exit('Error: Recheck '+str(optionset[idx][2])+' input\nPossible inputs: '+str(optionset[idx][0]) +' or '+str(optionset[idx][1])) 
   
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
@click.command()
@click.option('--data',nargs=1,type=str,default='fatigue_dataset.csv',help='Enter input dataset.csv: last column must be the target feature')
@click.option('--no_of_epoch',nargs=1,type=int,default=10001,help='Enter No of epoch for training(>0)')
@click.option('--pred_dataset',nargs=1,type=str,default='testset',help='Select dataset for prediction: [testset or trainingset]')
@click.option('--loss',nargs=1,type=str,default='MSE',help='Select loss: [MSE or RMSE]')
@click.option('--optimizer',nargs=1,type=str,default='SGD',help='Select optimizer: [SGD,SGDM,RMSP or Adam]')
@click.option('--layers',nargs=2,type=int,default=([10,10]),help='Enter hidden layer(N1,N2) input for [IP-N1-N2-OP](>0) based on HP tuning')
@click.option('--reg',nargs=6,type=float,default=([2e-6,2e-6,3e-6,3e-6,0,0]),help='Enter regularization loss(l2) for the layers')
@click.option('--sgd',nargs=2,type=float,default=([0.85,1e-1]),help='Enter SGD_optimizer input')
@click.option('--sgdm',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input')
@click.option('--rmsp',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input')
@click.option('--adam',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input')
@click.option('--makeplot',nargs=1,type=int,default=1,help='Select 0 or 1 to makeplot')
@click.option('--writeparam_as',nargs=1,type=str,default='npy',help='select write format: npy or txt')
@click.option('--writehl',nargs=1,type=int,default=0,help='Select 0 or 1 to write hl results')
@click.option('--predictor',nargs=1,type=str,default='OFF',help='Select ON or OFF for only prediction')
#
# ============================================================================================================================================
#                                                      CREATING MODEL --> ANN REGRESSION
# ============================================================================================================================================
#
def ANN_regression(data,no_of_epoch,pred_dataset,loss,optimizer,layers,reg,sgd,sgdm,rmsp,adam,makeplot,writeparam_as,writehl,predictor):
   '''
   ========================================================================
   Description:
   This ANN REGRESSION model can execute eight different combinations w.r.t loss and optimizer with one combination at a time.
   ========================================================================
   '''
   # Check certain user inputs
   #------------------------------------------------------------------------
   check_inputs(no_of_epoch,pred_dataset,loss,optimizer,layers,makeplot,writeparam_as,writehl,predictor)

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
   # Note: 
   # If predictor is ON all results of prediction w.r.t loss and optimizer inputs are present in the directory -> Results_ANN_predictor
   # The files present in the directory -> testingresults_lossinput_optimizerinput.txt and resultcomparision_lossinput_optimizerinput.txt
   #------------------------------------------------------------------------ 
   start_time1 = timeit.default_timer()                                     # Start timer to check the time taken by the model to execute
   model = ANN_Regression(data,no_of_epoch,hyp_paramANN,loss,optimizer,writehl,pred_dataset,predictor)

   # Train the model
   #------------------------------------------------------------------------ 
   if predictor == 'OFF':                                                   # If prediction is ON the model will not be trained  
      start_time2 = timeit.default_timer()                                  # Start timer to check the time taken for training the model 
      model.begin_training()
      timetaken_totrain = timeit.default_timer()-start_time2                # Stop timer. The time taken to train the model is noted; dtype -> float

      # Plot training results
      #------------------------------------------------------------------------
      if makeplot == 1:                                                     # If makeplot is satisfied(1) make the plots
         model.plot_trainingresults()
      
      # Write trained parameters
      #------------------------------------------------------------------------                        
      model.write_trainedparameters(writeparam_as)                          # Writing trained parameters
   else: 
      timetaken_totrain = 0                                                 # When predictor is ON training will not happen thus time taken to train wil be zero

   # Test the model
   #------------------------------------------------------------------------                           
   start_time3 = timeit.default_timer()                                     # Start timer to check the time taken for prediction in the model
   model.begin_prediction()
   timetaken_topred = timeit.default_timer()-start_time3                    # Stop time. The time taken to prediction in the model is noted; dtype -> float

   # Write data to make combination plots                                                                   
   #------------------------------------------------------------------------
   if predictor == 'OFF':                                                   # If prediction is ON plotting data will not be written and must not execute PPP_ANN_combination.py
      model.write_dataforPlot()

      # Plot prediction results and error
      #------------------------------------------------------------------------
      if makeplot == 1:                                                     # If makeplot is satisfied(1) make the plots
         model.plot_testingresults()
         model.plot_testingerror()

   # Write comparison data of the attained results with the original results
   #------------------------------------------------------------------------
   model.prediction_comparison()
   
   timetaken_permodel = timeit.default_timer()-start_time1                  # Stop time. The time taken by the model to execute is noted; dtype -> float

   # Write timing results for each combination -> Model, training and prediction
   #------------------------------------------------------------------------
   model.write_time(timetaken_permodel,timetaken_totrain,timetaken_topred)

   # Note: 
   # By including sigmoid activation function and categorical, binary cross entropy loss, this implementaton can be used for CNN & binary logistic regression

if __name__ == '__main__':
   ANN_regression()