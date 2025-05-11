'''
========================================================================
Test_file: test_MLR_Overfitting.py
Test_type: Overfitting test
Aim: To check the correctness of implementation the model is tested for overfitting
========================================================================
'''
# Import created libraries
# -------------------------
from PPP_MLR import Data_preprocessing
from PPP_MLR import Single_layer
from PPP_MLR import SGD_Optimizer
from PPP_MLR import MeanSquaredError_loss,RootMeanSquaredError_loss
from PPP_MLR import Accuracy

# Import required libraries
# -------------------------
import numpy as np
import sys
import click
#
# ======================================================================
# Construction of the model MLR Regression
# ======================================================================
#
class MLR_Regression:
    '''
    ========================================================================
    Description: 
    In this model class we compute coefficient of determination of the model after training and testing them.
    These results are then used to check whether the model is overfitting or not.
    ========================================================================
    '''
    def __init__(self,data,no_of_epoch,loss_input,hp):
        '''
        ========================================================================
        Description:
        Initializing the inputs and created libraries, performing data preprocessing steps and assigning single layer  
        ------------------------------------------------------------------------
        Parameters:
        data: Input dataset in .csv format with dependent feature as the last column 
        no_of_epoch: Training size; dtype -> int
        loss_input: Loss input; dtype -> str
        hp: Hyperparameters as dictionary
        ========================================================================
        '''
        self.no_of_epoch = no_of_epoch
        self.loss_input = loss_input 
        self.hp = hp 

        # Data_preprocessing
         # ------------------------------------------------------------------------
        Preprocessed_data = Data_preprocessing()      # Initialization for data preprocessing
        X,y = Preprocessed_data.import_dataset(data)  # X: Independent feature, y: dependent feature                  
        self.scaled_X,self.scaled_y,self.mean_y,self.std_y = Preprocessed_data.feature_scaling(X,y) # Scaled features, mean and std for rescaling during comparision
        self.X_train, self.X_test,self.y_train,self.y_test = Preprocessed_data.split_dataset(self.scaled_X,self.scaled_y) # Splitting dataset into training and test set
        self.X_train = np.hstack((np.ones((len(self.X_train),1)), self.X_train)) # Updating the training set to have ones as first column for MLR
        self.X_test = np.hstack((np.ones((len(self.X_test),1)), self.X_test))    # Updating the testing set to have ones as first row column MLR

        # Initialization for the single layer
        #------------------------------------------------------------------------
        self.single_layer =Single_layer(len(self.X_train[0]),self.hp['layers']['output_layer']) # Assigning layer from hp dictionary
        
        # Initialization for loss function and accuracy
        #------------------------------------------------------------------------
        self.loss_function_1 = MeanSquaredError_loss()
        self.loss_function_2 = RootMeanSquaredError_loss()
        self.model_accuracy = Accuracy()

        # Initialization for optimizers (Optimizer inputs are explained in PPP_MLR.py)
        #------------------------------------------------------------------------
        self.optimizer = SGD_Optimizer(self.hp['SGD']['learning_R'], self.hp['SGD']['learning_R_decay']) # Assigning optimizer inputs from hp dictionary

    def calculate_loss_Function(self,loss_Function,dataset):
        '''
        ========================================================================
        Description:
        To calculate data loss.
        ------------------------------------------------------------------------
        Parameters:
        loss_Function: Loss_function_1 or loss_function_2. loss calculation are performed based on the loss function
        dataset: Dependent feature(y_train or y_test); Array size -> [No of samples x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Return:
        loss: It is the data loss w.r.t predicted output and dependent feature; dtype -> float
        ========================================================================
        '''
        loss = loss_Function.calculate_loss(self.single_layer.output, dataset)
        return loss

    def calculate_back_prop(self,loss_Function,dataset):
        '''
        ========================================================================
        Description:
        To perform backward propagation in chain rule. 
        ------------------------------------------------------------------------
        Parameters:
        loss_Function: Loss_function_1 or loss_function_2. Different backward propagation procedure are fallowed w.r.t loss function input
        dataset: Dependent feature(y_train); Array size -> [No of samples x No of neurons required in the output layer]
        ------------------------------------------------------------------------
        Note:
        -The backward propagation goes from the loss function to the single layer
        -Backward propagation of single layer creates derivatives of weights and biases which are used in optimizers
        -More details on back propagation and its inputs can be found in PPP_MLR.py
        ========================================================================
        '''
        loss_Function.back_prop(self.single_layer.output, dataset)  # Back prop of loss function(MSE or RMSE)
        self.single_layer.back_prop(loss_Function.inputs_derv)      # Back prop of single layer

    def calculate_optimizer(self,optimizer):
        '''
        ========================================================================
        Description:
        To perform optimization and update parameters in the single layer.
        ------------------------------------------------------------------------
        Parameters:
        optimizer: SGD optimizer. Parameters update varies based on the optimizer input
        ------------------------------------------------------------------------
        Return:
        LearningR: Current learning rate from the optimizer; dtype -> float      
        ------------------------------------------------------------------------
        Note:
        -Derivatives of weights and biases which are used in optimizers to update weights and biases in the single layer
        -More details on optimizer and its inputs can be found in PPP_MLR.py
        ========================================================================
        '''
        optimizer.learning_R_update()                     # Update learning rate in the optimizer
        LearningR= optimizer.C_learning_R                 # Return current learning rate from the optimizer
        optimizer.parameters_update(self.single_layer)    # Update the parameter of single layer 
        optimizer.itr_update()                            # Update iteration step in the optimizer
        return LearningR
    
    def begin_training(self):
        '''
        ========================================================================
        Description:
        Training the model in a loop from forward propagation, backward propagation to optimization w.r.t the no of epoch.
        The loss and accuracy value of training result should be good enough so that the model can use the update weights and biases to make predictions.
        ------------------------------------------------------------------------
        Note:
        -Parameters are updated in each dense layer after training
        ========================================================================
        '''
        # Select loss from the input param dictionary
        #------------------------------------------------------------------------
        input_param = {'loss':{'MSE':self.loss_function_1, 'RMSE':self.loss_function_2}}

        # Select loss function based on loss input
        #------------------------------------------------------------------------
        if self.loss_input in input_param['loss'].keys():
             loss_function =  input_param['loss'][self.loss_input]    

        for epoch in range(self.no_of_epoch):  # Training begins w.r.t the no of epoch
        
            # Forward Propagation (More details on forward propagation and its inputs can be found in PPP_MLR.py)
            #------------------------------------------------------------------------
            self.single_layer.forward_prop(self.X_train)                      # Forward prop of single layer with independent feature as an input

            # Calculate Loss 
            #------------------------------------------------------------------------
            self.loss= self.calculate_loss_Function(loss_function,self.y_train)

            # Calculate accuracy (More details on coefficient_of_determination and its inputs can be found in PPP_MLR.py)
            #------------------------------------------------------------------------
            self.Rsqr = self.model_accuracy.coefficient_of_determination(self.single_layer.output, self.y_train)
            
            # Back Propagation 
            #------------------------------------------------------------------------
            self.calculate_back_prop(loss_function,self.y_train)

            # Perform Optimization
            #------------------------------------------------------------------------
            LearningR = self.calculate_optimizer(self.optimizer)

        # Forward prop for last optimizer update
        #-----------------------------------------------------------------------
        self.single_layer.forward_prop(self.X_train)

        # Calculate Loss for last optimizer update
        #------------------------------------------------------------------------
        self.loss= self.calculate_loss_Function(loss_function,self.y_train)

        # Calculate accuracy for last optimizer update
        #------------------------------------------------------------------------
        self.Rsqr = self.model_accuracy.coefficient_of_determination(self.single_layer.output, self.y_train)

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
       
        # Select dataset to perform prediction
        #------------------------------------------------------------------------
        self.pred_input = self.X_train           # In that case the trainined dataset X_train (independent features) will be the prediction input
        self.test_metric = self.y_train          # And y_train (dependent features) will be the testing metric
        
        # Forward Propagation (More details on forward propagation and its inputs can be found in PPP_MLR.py)
        #------------------------------------------------------------------------
        self.single_layer.forward_prop(self.pred_input)                   # Forward prop of single layer with independent feature as an input

        # Calculate Loss 
        #------------------------------------------------------------------------
        self.loss_pred = self.calculate_loss_Function(loss_function,self.test_metric) # The loss will be calculated again with the test metric

        # Calculate accuracy (More details on coefficient_of_determination and its inputs can be found in PPP_MLR.py)
        #------------------------------------------------------------------------
        self.Rsqr_pred = self.model_accuracy.coefficient_of_determination(self.single_layer.output, self.test_metric) # The acc will be calculated again with the test metric
        
def check_inputs(no_of_epoch,loss): 
    '''
    ========================================================================
    Description:
    To check whether certain user inputs are within the required limits.
    ------------------------------------------------------------------------
    Parameters:
    no_of_epoch: Training size; dtype -> int
    loss: Loss input; dtype -> str 
    ------------------------------------------------------------------------
    Note:
    -If the inputs are not within the options range, program exits by throwing an error with possible inputs
    ========================================================================
    '''
    # Checking whether the input is correct or wrong
    #-----------------------------------------------------------------------
    if (not loss == 'MSE') and (not loss == 'RMSE'):                           # Checking whether the give loss input is within the options range
        sys.exit('Error: Recheck loss input\nPossible inputs: MSE or RMSE')    # Else program exits by throwing an error mentioning possible inputs
    
    if(no_of_epoch <= 0):                                 # Checking epoch input, it should be greater than zero
        sys.exit('Error: no_of_epoch input must be > 0')  # Program exits if the input is lesser than or equal to zero
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
    @click.option('--sgd',nargs=2,type=float,default=([0.13,1e-1]),help='Enter SGD_optimizer input')
    #
    # ============================================================================================================================================
    #                                                      CREATING MODEL --> MLR REGRESSION
    # ============================================================================================================================================
    #
    def MLR_regression(data,no_of_epoch,loss,sgd):
        '''
        ========================================================================
        Description:
        This MLR REGRESSION model can be tested for two different combinations of loss and SGD optimizer input with one combination at a time.
        ========================================================================
        '''
        # Check certain user inputs
        #------------------------------------------------------------------------
        check_inputs(no_of_epoch,loss)

        # Adjustable hyperparameters must be tunned separately w.r.t the dataset used as it can impact the accuracy of the model
        # Creating the input dictionary of hp
        #------------------------------------------------------------------------    
        SGD_lr,SGD_lrd = sgd       # SGD hyperparameter inputs in form of tuple  @RMSE: SGD_lr = 0.5 so that the does not get stuck in local minima

        hyp_paramMLR = {'layers':{'output_layer':1},'SGD':{'learning_R':SGD_lr, 'learning_R_decay':SGD_lrd}}

        # Initializing the model
        #-----------------------------------------------------------------------
        model = MLR_Regression(data,no_of_epoch,loss,hyp_paramMLR)

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
        MLR_regression()

test_ANN_overfitting()