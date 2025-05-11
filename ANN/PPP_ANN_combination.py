'''
========================================================================
This program is a part of personal programming project
ANN program No: 3 
file: PPP_ANN_combination
------------------------------------------------------------------------
Creates comparison plots by reading in all the results created for eight combinations of loss and optimizer
Creates an comparison Excel of target and the predicted values for all the eight combinations of loss and optimizer
Creates a comparison table for the time taken by the model to perform a particular task at different instances for all combinations
These plots and datas are then used to analyse the model performance for different combinations
Condition1: This program should be executed only after executing PPP_ANN_createOverallresults.py @ predictor: OFF
Condition2: The input values should be same as the values that was used to create these datas by executing PPP_ANN_createOverallresults.py
'''
# Import created libraries
# -------------------------
from PPP_ANN import Data_preprocessing
from PPP_ANN import MeanSquaredError_loss,RootMeanSquaredError_loss
from PPP_ANN import Accuracy

# Import required libraries
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import click
#
# ======================================================================
# Creating comparison plots and datas for all the combinations of ANN Regression model 
# ======================================================================
#
class Result_comparison:
    '''
    ========================================================================
    Description: 
    Creating comparison plots and datas from the results attained from different combinations of loss and optimizer in ANN regression.
    Making comparison plots for loss, accuracy results and learning rate updates during training, plots for overall testing results, 
    comparison plots for different hidden layer combinations and finally model performance analysis plots for ANN.
    Making comparison Excels for prediciton results and tabulating the time taken results for all the combinations.
    ========================================================================
    '''

    def __init__(self,data,no_of_epoch,ls,op,losslist,optimizerlist,hp,hl,limits): 
        '''
        ========================================================================
        Description:
        Initializing the inputs and created libraries, performing data preprocessing steps and 
        creating storage to store data.  
        ------------------------------------------------------------------------
        Parameters:
        data: Input dataset in .csv format with dependent feature as the last column
        no_of_epoch: Training size; dtype -> int
        ls: Loss input from the best performing combination and the same is used to create hidden layer combinations; dtype -> str
        op: Optimizer input from the best performing combination and the same is used to create hidden layer combinations; dtype -> str
        losslist: Loss input list; dtype -> str
        optimizerlist: Optimizer input list; dtype -> str
        hp: Hyperparameters as dictionary
        hl: Hidden layer combinations list; dtype -> int
        limits: Its a tuple used for selecting certain ranges of data for plotting loss,accuracy, learning rate and HL plot; dtype -> int
        ========================================================================
        '''
        self.No_of_epoch = no_of_epoch
        self.loss = losslist
        self.optimizer = optimizerlist
        self.hp,self.hl = hp,hl
        self.ls,self.op = ls,op
        self.limit,self.sublimit,self.limit1,self.sublimit1,self.limit2,self.sublimit2 = limits
        
        # Initialization for Data_preprocessing, loss_function and accuracy
        #------------------------------------------------------------------------
        Preprocessed_data = Data_preprocessing()
        self.loss_function_1 = MeanSquaredError_loss()
        self.loss_function_2 = RootMeanSquaredError_loss()
        self.accuracy = Accuracy()

        # Data_preprocessing
        # ------------------------------------------------------------------------
        X,y = Preprocessed_data.import_dataset(data)                                                             # X: Independent feature, y: dependent feature         
        scaled_X,scaled_y,self.mean_y,self.std_y = Preprocessed_data.feature_scaling(X,y)                        # Scaled features, mean and std for rescaling during comparison
        self.X_train, self.X_test,self.y_train,self.y_test = Preprocessed_data.split_dataset(scaled_X,scaled_y)  # Splitting dataset into training and test set
       
        # Creating Storage to store read in datas of different combinations to make plots and data comparisions
        #------------------------------------------------------------------------
        self.overall_loss = []                                         # To store overall data loss
        self.overall_learningR = []                                    # To store overall learning rate update
        self.overall_coef_det = []                                     # To store overall accuracy
        self.overall_testingResult = []                                # To store overall testing results
        self.hiddenlayer_loss = []                                     # To store losses from hidden layer combinations
        self.hiddenlayer_testResult = []                               # To store test results from hidden layer combinations
        self.time_hl = []                                              # To store overall time taken by different combinations of hidden layer to execute
        self.timefw,self.timebw,self.timeop = [],[],[]                 # To store overall time taken by forward prop, backward prop and optimization to execute during training
        self.timemod,self.timetrain,self.timepred = [],[],[]           # To store overall time taken by model, training and testing to execute
        self.timecode = []                                             # To store overall time taken by each code to execute
        
    def read_data(self):
        '''
        ========================================================================
        Description:
        To read in data from all the combination to make comparison plots.
        Reading in loss, learning rate ,accuracy data and testing results from the file: plotdata_lossinput_optimizerinput.txt in the directory: Results_forPlotting.
        Reading in hiddenlayer combination results from the file: lossatHiddLayHL1HL2OL3_lossinput_optimizerinput.txt in the same directory.
        Reading in the time results from the file1: resultcomparison_lossinput_optimizerinput.txt and file2: timetaken_overall.txt in the directory: Results_TargetVSpred.
        ========================================================================
        '''
        for ls_ip in self.loss:
            for op_ip in self.optimizer:
                # Creating paths and filenames to open
                #------------------------------------------------------------------------
                open_path = os.path.abspath('Results_forPlotting')                                     # Creating path1 to open directory -> Results_forPlotting 
                open_path1 = os.path.abspath('Results_TargetVSpred')                                   # Creating path2 to open directory -> Results_TargetVSpred
                filename = os.path.join(open_path,'plotdata_%s_%s.txt' % (ls_ip,op_ip))                # File1 to open -> plotdata_lossinput_optimizerinput.txt
                filename1 = os.path.join(open_path1, 'resultcomparison_%s_%s.txt' % (ls_ip,op_ip))     # File2 to open -> resultcomparision_lossinput_optimizerinput.txt
                filename2 = os.path.join(open_path1,'timetaken_overall.txt')                           # File3 to open -> timetaken_overall.txt
                
                # Creating temporary storage to append the results of each combination as a list
                #------------------------------------------------------------------------
                Temp_loss = []                                                          # Temporary loss storage
                Temp_learningR = []                                                     # Temporary learning rate storage
                Temp_coef_det = []                                                      # Temporary accuracy storage
                Temp_testingResult = []                                                 # Temporary testing results storage

                # Reading results from plotdata_lossinput_optimizerinput.txt to plot
                #------------------------------------------------------------------------  
                with open(filename, 'r') as f:                                          
                    read_data = f.readlines()                                           # Reading in all the results 
            
                    for idx in (np.arange(7,2*self.No_of_epoch+7,2)):
                        read_loss = float(read_data[idx].split('|')[1])                 # Reading in loss results of training
                        read_learningR = float(read_data[idx].split('|')[2])            # Reading in learning rate updates of training
                        read_coef_det = float(read_data[idx].split('|')[3])             # Reading in accuracy results of training
                        
                        Temp_loss.append(read_loss)                                     # Storing loss results of each combination
                        Temp_learningR.append(read_learningR)                           # Storing learning rate update of each combination
                        Temp_coef_det.append(read_coef_det)                             # Storing accuracy results of each combination
                    
                    for idx in (np.arange(7,2*len(self.y_test)+7,2)):                   
                        read_testingResult = float(read_data[idx].split('|')[4])        # Reading in testing results
                        
                        Temp_testingResult.append(read_testingResult)                   # Storing testing results of each combination

                self.overall_loss.append(Temp_loss)                                     # Storing loss results of each combination in overall loss
                self.overall_learningR.append(Temp_learningR)                           # Storing learning rate update of each combination in overall learning rate
                self.overall_coef_det.append(Temp_coef_det)                             # Storing accuracy results of each combination in overall accuracy
                self.overall_testingResult.append(Temp_testingResult)                   # Storing testing results of each combination in overall testing results

                # Reading results from resultcomparision_lossinput_optimizerinput.txt to make time comparison table
                #------------------------------------------------------------------------
                with open(filename1, 'r') as f:
                    read_data = f.readlines()                                           # Reading time taken results
                    readfw = read_data[2*len(self.y_test)+14].split(';')[0]             # Reading in time taken for forward prop during training
                    readbw = read_data[2*len(self.y_test)+14].split(';')[1]             # Reading in time taken for backward prop during training
                    readop = read_data[2*len(self.y_test)+14].split(';')[2]             # Reading in time taken for Optimization during training
                    readmod = read_data[2*len(self.y_test)+15].split(';')[0]            # Reading in time taken by the model to execute
                    readtrain = read_data[2*len(self.y_test)+15].split(';')[1]          # Reading in time taken for training the model
                    readpred = read_data[2*len(self.y_test)+15].split(';')[2]           # Reading in time taken for predicting the model

                    self.timefw.append(float(readfw.split('=')[1]))                     # Storing time taken for forward prop during training for each combination
                    self.timebw.append(float(readbw.split('=')[1]))                     # Storing time taken for backward prop during training for each combination
                    self.timeop.append(float(readop.split('=')[1]))                     # Storing time taken for Optimization during training for each combination
                    self.timemod.append(float(readmod.split('=')[1]))                   # Storing time taken by the model for each combination
                    self.timetrain.append(float(readtrain.split('=')[1]))               # Storing time taken for training the model for each combination
                    self.timepred.append(float(readpred.split('=')[1]))                 # Storing time taken for predicting the model for each combination

        # Reading results from timetaken_overall.txt to make time comparison table
        #------------------------------------------------------------------------
        with open(filename2, 'r') as f:
            read_data = f.readlines()                                                   # Reading time taken results
            for idx in (np.arange(7,2*len(self.loss)*len(self.optimizer)+7,2)):
                self.timecode.append(float(read_data[idx].split('|')[2]))               # Storing time taken for each code to execute in PPP_ANN_createOverallresults.py
            self.timeall = float(read_data[2*len(self.loss)*len(self.optimizer)+8].split(':')[1]) # Time taken for all the eight combination to execute in PPP_ANN_createOverallresults.py

        for hl_ip in self.hl:
            # Creating temporary storage to append the results of each combination as a list
            #------------------------------------------------------------------------
            Temp_HLloss = []                                                            # Temporary HL loss storage
            Temp_HLtestingResult =  []                                                  # Temporary HL testing results storage

            # Reading results from lossatHiddLayHL1HL2OL3_lossinput_optimizerinput.txt to plot
            #------------------------------------------------------------------------
            filename3 = os.path.join(open_path,'lossatHiddLay%d%d%d_%s_%s.txt' % (hl_ip[0],hl_ip[1],1,self.ls,self.op))
            with open(filename3, 'r') as f:

                read_HLdata = f.readlines()                                             # Reading in all the results
                for idx in (np.arange(7,2*self.No_of_epoch+7,2)):
                    read_HLloss = float(read_HLdata[idx].split('|')[1])                 # Reading in HL loss results of training
                    Temp_HLloss.append(read_HLloss)                                     # Storing HL loss results of each HL combination
                for idx in (np.arange(7,2*len(self.y_test)+7,2)):
                    read_HLtestRes = float(read_HLdata[idx].split('|')[2])              # Reading in HL testing results 
                    Temp_HLtestingResult.append(read_HLtestRes)                         # Storing HL testing results of each HL combination
                read_time = float(read_HLdata[2*self.No_of_epoch+8].split(' ')[5])      # Reading in the time taken by each HL combination to execute
             
            self.hiddenlayer_loss.append(Temp_HLloss)                                   # Storing HL loss results of each HL combination in overall HL loss
            self.hiddenlayer_testResult.append(Temp_HLtestingResult)                    # Storing HL testing results of each HL combination in overall HL testing results
            self.time_hl.append(read_time)                                              # Storing time taken by each HL combination to execute
        
    def calculate_lossandRsqr(self,index):
        '''
        ========================================================================
        Description:
        To calculate data loss at MSE and RMSE and to calculate accuracy.
        ------------------------------------------------------------------------
        Parameters:
        index: Selecting results from the 'list of results' to perform calculations w.r.t the loss and optimizer combination; dtype -> int
        ------------------------------------------------------------------------
        Note:
        -Combinations w.r.t the index = [ If 0: MSE_SGD, 1: MSE_SGDM, 2: MSE_RMSP, 3: MSE_Adam, 4: RMSE_SGD, 5: RMSE_SGDM, 6: RMSE_RMSP, 7: RMSE_Adam]
        ========================================================================
        '''
        self.loss_MSE = self.loss_function_1.calculate_loss(np.array([self.overall_testingResult[index]]).T, self.y_test)    # Calculating MSE loss based on result selection
        self.loss_RMSE = self.loss_function_2.calculate_loss(np.array([self.overall_testingResult[index]]).T, self.y_test)   # Calculating RMSE loss based on result selection 
        self.Rsqr = self.accuracy.coefficient_of_determination(np.array([self.overall_testingResult[index]]).T, self.y_test) # Calculating accuracy based on result selection
        
    def write_TargetVSpred(self):
        '''
        ========================================================================
        Description:
        To create a comparison Excel between the target results and the predicted results for all the combinations of loss and optimizer. 
        From this Excel we can compare the predicted results of all the combinations for each sample(line item) against the target results and perform result analysis.
        ------------------------------------------------------------------------
        Conditions:
        make sure xlsxwriter is available or pip install xlsxwriter
        ------------------------------------------------------------------------
        Output:
        Writes a file: resultcomparisonANN.xlsx to the directory: Results_TargetVSpred         
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_TargetVSpred'):
            os.makedirs('Results_TargetVSpred')
        save_path = os.path.abspath('Results_TargetVSpred')  # Save the file to the created directory
       
        # Creating inputs for each column of the dataframe from the stored prediction results
        #------------------------------------------------------------------------
        c1,c2, = (self.y_test*self.std_y + self.mean_y).reshape(-1,), (self.overall_testingResult[0]*self.std_y + self.mean_y).reshape(-1,)
        c3,c4 = (self.overall_testingResult[1]*self.std_y + self.mean_y).reshape(-1,), (self.overall_testingResult[2]*self.std_y + self.mean_y).reshape(-1,)
        c5,c6 = (self.overall_testingResult[3]*self.std_y + self.mean_y).reshape(-1,), (self.overall_testingResult[4]*self.std_y + self.mean_y).reshape(-1,)
        c7,c8 = (self.overall_testingResult[5]*self.std_y + self.mean_y).reshape(-1,), (self.overall_testingResult[6]*self.std_y + self.mean_y).reshape(-1,)
        c9 = (self.overall_testingResult[7]*self.std_y + self.mean_y).reshape(-1,)

        # Creating a dataframe to write results in a tablulated xlsx format
        #------------------------------------------------------------------------
        writedata = pd.DataFrame({'Target': c1,'Pred_MSE_SGD': c2,'Pred_MSE_SGDM': c3,'Pred_MSE_RMSP': c4,'Pred_MSE_Adam': c5,'Pred_RMSE_SGD': c6,\
                                  'Pred_RMSE_SGDM': c7,'Pred_RMSE_RMSP': c8,'Pred_RMSE_Adam': c9})           # Assigning column header
        # Writing an Excel
        #------------------------------------------------------------------------
        writer = pd.ExcelWriter(os.path.join(save_path, 'resultcomparisonANN.xlsx'))                         # Creating an Excel writer 
        writedata.to_excel(writer, sheet_name='targetVSpred',float_format = '%.0f',index = False,startrow=1) # Converting dataframe into an Excel by specifying the sheet name, 
                                                                                                             # floating format, starting row and avoiding dataframe index
        # Getting xlsxwriter objects from the dataframe writer object
        #------------------------------------------------------------------------                                                                                                     
        getworkbook = writer.book                                                                            
        getworksheet = writer.sheets['targetVSpred']                                                         # Specify the sheet name

        writetitle = 'Result comparison: Target vs Predicted for all combinations(ANN)'                      # Setting title for the Excel
        getworksheet.write(0,0,writetitle)                                                                   # Writing the title in the Excel at a specified location
        cellformat_title = getworkbook.add_format({'bold':True,'font_size': 20})                             # Formating the title by making it bold and setting a font size 
        getworksheet.set_row(0,40,cellformat_title)                                                          # Applying the formating and setting the title row height
    
        cellformat_border = getworkbook.add_format({'border':1})                                             # Setting closed borders to the Excel                                   
        getworksheet.conditional_format('A1:I%d'%(len(self.y_test)+2), {'type':'no_blanks','format':cellformat_border}) # Applying boarder settings for required rows and cols
        cellformat_column = getworkbook.add_format({'align':'center'})                                       # Setting center alignment to the datas
        getworksheet.set_column('A1:I%d'%(len(self.y_test)+2), 16,cellformat_column)                         # Applying the alignment setting and adjusting the column length
        
        # Using dataframe to calculate the mean and the std of each combination and writing it in the Excel
        #------------------------------------------------------------------------
        mean,std = np.array(writedata.mean()),np.array(writedata.std())                                      # Calculating mean and std                                   
        writemean = pd.DataFrame({'Target': [mean[0]],'Pred_MSE_SGD': [mean[1]],'Pred_MSE_SGDM': [mean[2]],'Pred_MSE_RMSP': [mean[3]],'Pred_MSE_Adam': [mean[4]],\
                                  'Pred_RMSE_SGD': [mean[5]],'Pred_RMSE_SGDM': [mean[6]],'Pred_RMSE_RMSP': [mean[7]],'Pred_RMSE_Adam': [mean[8]]})
        writestd = pd.DataFrame({'Target': [std[0]],'Pred_MSE_SGD': [std[1]],'Pred_MSE_SGDM': [std[2]],'Pred_MSE_RMSP': [std[3]],'Pred_MSE_Adam': [std[4]],\
                                  'Pred_RMSE_SGD': [std[5]],'Pred_RMSE_SGDM': [std[6]],'Pred_RMSE_RMSP': [std[7]],'Pred_RMSE_Adam': [std[8]]})
        writemean.to_excel(writer, sheet_name='targetVSpred',float_format = '%.3f',header = False,index = False,startrow=len(self.y_test)+3)# Converting dataframe into an Excel
        writestd.to_excel(writer, sheet_name='targetVSpred',float_format = '%.3f',header = False,index = False,startrow=len(self.y_test)+4) # Converting dataframe into an Excel

        bold = getworkbook.add_format({'bold': 1})                                                           # Setting bold formating 
        getworksheet.write(len(self.y_test)+3,9,'Mean',bold)                                                 # Writing the mean in the Excel at a specified location
        getworksheet.write(len(self.y_test)+4,9,'Std',bold)                                                  # Writing the std in the Excel at a specified location

        # Adding Notes
        #------------------------------------------------------------------------
        note ='Note: Pred_x_y are predicted values using x: loss and y: optimizer'                           # Setting additional note
        getworksheet.write(len(self.y_test)+6,0,note)                                                        # Writing the note in the Excel at a specified location
        cellformat_note = getworkbook.add_format({'bold':True,'font_size': 15})                              # Formating the note by making it bold and setting a font size
        getworksheet.set_row(len(self.y_test)+6,20,cellformat_note)                                          # Applying the formating and setting the title row height
        writer.save()                                                                                        # Close the Excel writer and output the Excel file

    def write_time(self):
        '''
        ========================================================================
        Description:
        To create a comparison table that compares the time taken by the model at different instances for all the eight combinations. 
        ------------------------------------------------------------------------
        Conditions:
        Make sure tabulate package is installed or pip install tabulate
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Results_Timetaken and writes a file: resulttimetakenANN.txt
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_Timetaken'):
            os.makedirs('Results_Timetaken')
        save_path = os.path.abspath('Results_Timetaken') # Save the file to the created directory

        # Creating a dataframe for time taken results so that the results can be written in a tablulated form
        #------------------------------------------------------------------------
        comb = ['MSE_SGD','MSE_SGDM','MSE_RMSP','MSE_Adam','RMSE_SGD','RMSE_SGDM','RMSE_RMSP','RMSE_Adam'] # All the eight combinations list to tabulate
        writedata = pd.DataFrame({'Combinations': comb,'Code':self.timecode,'Model':self.timemod,'Training': self.timetrain,'Prediction': self.timepred,\
            'Forward_prop': self.timefw,'Backward_prop': self.timebw,'Optimizer': self.timeop})

        # writing time taken results into the file -> resulttimetakenANN.txt
        #------------------------------------------------------------------------
        with open(os.path.join(save_path, 'resulttimetakenANN.txt'), 'w') as f:

            print('Time taken in seconds by ANN at different instances and combinations:\n',file=f) # Creating title infos before writing
            print(writedata.to_markdown(tablefmt='grid',floatfmt='.6f',index=False),file=f)         # Tabulating the results in grid format without index
            print('\nTime taken by all the eight combination to execute:',self.timeall,file=f)      # Writing results of time taken for all the combinations

    def write_Modelperformance(self):
        '''
        ========================================================================
        Description:
        To write loss and accuracy values for all the combinations of ANN model.
        ------------------------------------------------------------------------
        Conditions:
        Make sure tabulate package is installed or pip install tabulate
        ------------------------------------------------------------------------
        Output:
        Creates a directory if one does not exists: Results_TargetVSpred and writes a file: resultModelperformance_ANN.txt  
        ========================================================================
        '''
        training_acc_MSE, training_acc_RMSE= [], []                                                 # Creating storage to store training accuracy                                                                
        training_loss_MSE, training_loss_RMSE = [], []                                              # Creating storage to store training loss

        for idx in range(len(self.optimizer)):                                                      
            training_acc_MSE.append(self.overall_coef_det[idx][self.No_of_epoch-1])                 # Storing training accuracy created with MSE loss
            training_acc_RMSE.append(self.overall_coef_det[idx+4][self.No_of_epoch-1])              # Storing training accuracy created with RMSE loss
            training_loss_MSE.append(self.overall_loss[idx][self.No_of_epoch-1])                    # Storing training loss created with MSE loss
            training_loss_RMSE.append(self.overall_loss[idx+4][self.No_of_epoch-1])                 # Storing training loss created at RMSE loss
        
        training_acc = training_acc_MSE + training_acc_RMSE                                         # Combining training accuracies computed at MSE and RMSE loss
        training_loss = training_loss_MSE + training_loss_RMSE                                      # Combining training losses computed at MSE and RMSE loss
        pred_acc = self.store_coef_detMSE + self.store_coef_detRMSE                                 # Combining prediction accuracies computed at MSE and RMSE loss
        pred_loss = self.store_MSE_ls + self.store_RMSE_ls                                          # Combining prediction losses computed at MSE and RMSE loss
        acc_diff = np.array(pred_acc) - np.array(training_acc)                                      # Computing the accuracy difference between the training and prediction
        loss_diff = np.array(pred_loss) - np.array(training_loss)                                   # Computing the loss difference between the training and prediction

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_TargetVSpred'):
            os.makedirs('Results_TargetVSpred')
        save_path = os.path.abspath('Results_TargetVSpred') # Save the file to the created directory

        # Creating a dataframe so that the results can be written in a tablulated form
        #------------------------------------------------------------------------
        comb = ['MSE_SGD','MSE_SGDM','MSE_RMSP','MSE_Adam','RMSE_SGD','RMSE_SGDM','RMSE_RMSP','RMSE_Adam'] # All the eight combinations list to tabulate
        writedata = pd.DataFrame({'Combinations': comb,'Training\naccuracy':training_acc,'Validation\naccuracy':pred_acc,'Accuracy\ndifference': acc_diff,\
            'Training\nloss': training_loss,'Validation\nloss': pred_loss,'Loss\ndifference': loss_diff})

        # writing time taken results into the file -> resultModelperformance_ANN.txt
        #------------------------------------------------------------------------
        with open(os.path.join(save_path, 'resultModelperformance_ANN.txt'), 'w') as f:

            print('ANN model results for all combinations during training and validation:\n',file=f) # Creating title infos before writing
            print(writedata.to_markdown(tablefmt='grid',floatfmt='.6f',index=False),file=f)          # Tabulating the results in grid format without index

    def plot_OverallLoss(self):
        '''
        ========================================================================
        Description:
        Creating a overall loss plot for all the combination at MSE and RMSE loss.
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Plots_ANN_combination and saves the plot: plot_Overall_Loss.png
        ------------------------------------------------------------------------
        Note:
        -The user input limits is used to set x-limits and are used to select the range of values for loss plots
        -In the overall loss plot the first two entries of the limits tuple is considered as limit and sublimit
        ========================================================================
        '''
        # Plot for overall loss
        #------------------------------------------------------------------------
        ax1 = plt.subplot2grid((1,2),(0,0))                                          # Plotting in grids -> first row first column @MSE
        ax1.set_title('Optimization algorithms vs MSE loss')                         # Set plot title
        ax1.set_xlabel('No of epoch(x%d)'%np.round(self.No_of_epoch/self.limit))     # Set label for x-axis -> will be the no of epoch with reduced magnitude
        ax1.set_ylabel('Loss value')                                                 # Set label for y-axis -> will be overall loss values of training
        ax1.set_xlim(0,self.limit)                                                   # limits, sublimits are based on the user input tuple -> limits[0], limits[1]
        self.linestyle = ['o-','*-','^-','s-']                                       # Setting line styles w.r.t the no of optimizers
        for index,optimizer in enumerate(self.optimizer):
            ax1.plot(np.arange(0,self.limit,self.sublimit),self.overall_loss[index][:self.limit][::self.sublimit],self.linestyle[index],\
                label = str(optimizer)+'; ls = %.4f'% self.overall_loss[index][self.No_of_epoch-1]) # Plotting overall MSE loss and the values are accessed from the list using index
        ax1.legend(loc='upper right')
        
        ax2 = plt.subplot2grid((1,2),(0,1))                                          # Plotting in grids -> first row second column @RMSE
        ax2.set_title('Optimization algorithms vs RMSE loss')                        # Set plot title
        ax2.set_xlabel('No of epoch(x%d)'%np.round(self.No_of_epoch/self.limit))     # Set label for x-axis -> will be the no of epoch with reduced magnitude
        ax2.set_ylabel('Loss value')                                                 # Set label for y-axis -> will be overall loss values of training
        ax2.set_xlim(0,self.limit)                                                   # limits are based on the user input tuple -> limits[0]
        for index,optimizer in enumerate(self.optimizer):
            ax2.plot(np.arange(0,self.limit,self.sublimit),self.overall_loss[index+4][:self.limit][::self.sublimit],self.linestyle[index],\
                 label = str(optimizer)+'; ls = %.4f'% self.overall_loss[index+4][self.No_of_epoch-1]) # Plotting overall RMSE loss and the values are accessed from the list using index
        ax2.legend(loc='upper right')

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_ANN_combination'):
            os.makedirs('Plots_ANN_combination') 
        save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory

        fig = plt.gcf()                                                                     # Get the current figure
        fig.set_size_inches((15.5, 5),forward = False)                                      # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                    # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path, 'plot_Overall_Loss.png'),bbox_inches='tight')   # Saving plot to the specified path
        plt.clf()                                                                           # Cleaning the current figure after saving
    
    def plot_OverallAccuracy(self):
        '''
        ========================================================================
        Description:
        Creating a overall accuracy plot for all the combination at MSE and RMSE loss.
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_ANN_combination and saves the plot: plot_Overall_Accuracy.png
        ------------------------------------------------------------------------
        Note:
        -Similar to loss plots the user input limits are also used in accuracy plots
        -The limit and sublimit are the same for overall loss plot and overall accuracy plot
        ========================================================================
        '''
        # Plot for accuracy
        #------------------------------------------------------------------------
        ax1 = plt.subplot2grid((1,2),(0,0))                                         # Plotting in grids -> first row first column @MSE
        ax1.set_title('Optimization algorithms vs Accuracy (@MSE)')                 # Set plot title
        ax1.set_xlabel('No of epoch(x%d)'%np.round(self.No_of_epoch/self.limit))    # Set label for x-axis -> will be the no of epoch with reduced magnitude
        ax1.set_ylabel('$R^2$ value')                                               # Set label for y-axis -> will be overall accuracy values of training
        ax1.set_xlim(0,self.limit)                                                  # limits, sublimits are based on the user input tuple -> limits[0], limits[1]
        for index,optimizer in enumerate(self.optimizer):
            ex_label =  str(optimizer)+'; $R^2$ = %.3f'% self.overall_coef_det[index][self.No_of_epoch-1]  # Creating label for accuracy
            ax1.plot(np.arange(0,self.limit,self.sublimit),self.overall_coef_det[index][:self.limit][::self.sublimit],self.linestyle[index], label = ex_label)
        ax1.legend(loc='lower right')                                               # Plotting overall accuracy @MSE loss and the values are accessed from the list using index

        ax2 = plt.subplot2grid((1,2),(0,1))                                         # Plotting in grids -> first row second column @RMSE
        ax2.set_title('Optimization algorithms vs Accuracy (@RMSE)')                # Set plot title
        ax2.set_xlabel('No of epoch(x%d)'%np.round(self.No_of_epoch/self.limit))    # Set label for x-axis -> will be the no of epoch with reduced magnitude
        ax2.set_ylabel('$R^2$ value')                                               # Set label for y-axis -> will be overall accuracy values of training
        ax2.set_xlim(0,self.limit)                                                  # limits, sublimits are based on the user input tuple -> limits[0], limits[1]
        for index,optimizer in enumerate(self.optimizer):
            ex_label = str(optimizer)+'; $R^2$ = %.3f'% self.overall_coef_det[index+4][self.No_of_epoch-1] # Creating label for the accuracy plot
            ax2.plot(np.arange(0,self.limit,self.sublimit),self.overall_coef_det[index+4][:self.limit][::self.sublimit],self.linestyle[index], label = ex_label)
        ax2.legend(loc='lower right')                                               # Plotting overall accuracy @RMSE loss and the values are accessed from the list using index

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_ANN_combination'):
            os.makedirs('Plots_ANN_combination') 
        save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory

        fig = plt.gcf()                                                                     # Get the current figure
        fig.set_size_inches((15.5,5),forward = False)                                       # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                    # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path, 'plot_Overall_Accuracy.png'),bbox_inches='tight') # Saving plot to the specified path
        plt.clf()                                                                           # Cleaning the current figure after saving

    def plot_OveralllearningR(self):
        '''
        ========================================================================
        Description:
        Creating a relation plot between overall learning rate update and loss values for SGD and SGDM at MSE and RMSE loss.
        As the learning rate decreases the loss value also decreases.
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_ANN_combination and saves the plot: plot_Overall_LearningRate.png
        ------------------------------------------------------------------------
        Note:
        -Similar to loss plots the user input limits are also used in learning rate update plots
        -In the learning rate update plot the third and fourth entries of the limits tuple is considered as limit1 and sublimit1
        ========================================================================
        '''
        # Plot for learning rate
        #------------------------------------------------------------------------
        ax1 = plt.subplot2grid((1,2),(0,0))                                         # Plotting in grids -> first row first column @MSE
        ax1.set_title('Learning rate vs MSE loss')                                  # Set plot title
        ax1.set_xlabel('Learning rate value')                                       # Set label for x-axis -> will be the learning rate value
        ax1.set_ylabel('Loss value')                                                # Set label for y-axis -> will be overall loss values of training
        self.linestyle1 = ['o-','*-']                                               # Setting line styles for SGD and SGDM optimizer
        for index,optimizer in enumerate(self.optimizer[:2]):
            # Create label at SGD and SGDM optimizers
            #------------------------------------------------------------------------
            if optimizer == 'SGD': 
                extend = ' lr_d = '+str(self.hp[optimizer]['learning_R_decay'])
            elif optimizer == 'SGDM':
                extend = ' lr_d = '+str(self.hp[optimizer]['learning_R_decay'])+'; m = '+str(self.hp[optimizer]['momentum'])      
            ex_label = str(optimizer)+': lr = %.2f;'%(self.overall_learningR[index][0])+str(extend)

            ax1.plot(self.overall_learningR[index][:self.limit1][::self.sublimit1],self.overall_loss[index][:self.limit1][::self.sublimit1],self.linestyle1[index],\
                label = ex_label)                   # Plotting relation plot between learning rate and MSE loss and the values are accessed from the list using index
        ax1.invert_xaxis()                                                          # Inverting the axis to show how the learning rate decrease with decrease in loss
        ax1.legend(loc='lower left')
        
        ax2 = plt.subplot2grid((1,2),(0,1))                                         # Plotting in grids -> first row second column @RMSE
        ax2.set_title('Learning rate vs RMSE loss')                                 # Set plot title
        ax2.set_xlabel('Learning rate value')                                       # Set label for x-axis -> will be the learning rate value
        ax2.set_ylabel('Loss value')                                                # Set label for y-axis -> will be overall loss values of training
        for index,optimizer in enumerate(self.optimizer[:2]):
            # Create label at SGD and SGDM optimizers
            #------------------------------------------------------------------------
            if optimizer == 'SGD': 
                extend = ' lr_d = '+str(self.hp['SGD_RMSE']['learning_R_decay'])
            elif optimizer == 'SGDM':
                extend = ' lr_d = '+str(self.hp['SGDM_RMSE']['learning_R_decay'])+'; m = '+str(self.hp['SGDM_RMSE']['momentum'])      
            ex_label = str(optimizer)+': lr = %.3f;'%(self.overall_learningR[index+4][0])+str(extend)

            ax2.plot(self.overall_learningR[index+4][:self.limit1][::self.sublimit1],self.overall_loss[index+4][:self.limit1][::self.sublimit1],self.linestyle1[index],\
                label = ex_label)                  # Plotting relation plot between learning rate and RMSE loss and the values are accessed from the list using index
        ax2.invert_xaxis()                                                          # Inverting the axis to show how the learning rate decrease with decrease in loss
        ax2.legend(loc='lower left')

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_ANN_combination'):
            os.makedirs('Plots_ANN_combination') 
        save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory
        
        fig = plt.gcf()                                                                     # Get the current figure
        fig.set_size_inches((15.5,5),forward = False)                                       # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                    # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path, 'plot_Overall_LearningRate.png'),bbox_inches='tight') # Saving plot to the specified path
        plt.clf()                                                                           # Cleaning the current figure after saving
    
    def plot_hypHiddenlayer(self):
        '''
        ========================================================================
        Description:
        Creating a comparison plot for different hiddenlayer(HL) combinations and mentioning the time taken and the accuracy of the different HL combinations.
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_ANN_combination and saves the plot: plot_LossatHL.png
        ------------------------------------------------------------------------
        Note:
        -Similar to loss plots the user input limits are also used in hidden layer comparison plots
        -In the hidden layer comparison plot the fifth and last entries of the limits tuple is considered as limit2 and sublimit2
        ========================================================================
        '''
        plt.title('Hidden layers[IP-N1-N2-OP] vs %s loss (@%s)'% (self.ls,self.op)) # Set plot title w.r.t the HL combinations
        plt.xlabel('No of epoch(x%d)'%np.round(self.No_of_epoch/self.limit2))       # Set label for x-axis -> will be the no of epoch with reduced magnitude
        plt.ylabel('Loss value')                                                    # Set label for y-axis -> will be loss values of HL
        for index,hl in enumerate(self.hl):
            loss = self.loss_function_1.calculate_loss(np.array([self.hiddenlayer_testResult[index]]).T, self.y_test) # Calculating loss for the HL testing results
            Rsqr = self.accuracy.coefficient_of_determination(np.array([self.hiddenlayer_testResult[index]]).T, self.y_test) # Calculating accuracy for the HL testing results
            plt.plot(np.arange(0,self.limit2,self.sublimit2),self.hiddenlayer_loss[index][:self.limit2][::self.sublimit2],self.linestyle[index],\
                label = '%d-%d-%d-%d; $R^2$ = %.3f; t = %.3fs'%(len(self.X_train[0]),hl[0],hl[1],1,Rsqr,self.time_hl[index])) # Plotting comparison and values are accessed using index
        plt.legend()

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_ANN_combination'):
            os.makedirs('Plots_ANN_combination') 
        save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory
       
        fig = plt.gcf()                                                                     # Get the current figure
        fig.set_size_inches((7.5,5),forward = False)                                        # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                    # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path, 'plot_LossatHL.png'),bbox_inches='tight')       # Saving plot to the specified path
        plt.clf()                                                                           # Cleaning the current figure after saving

    def plot_OverallResult(self,targetFeature,Acclim,lslim):
        '''
        ========================================================================
        Description:
        Creating an analysis plot comparing the performance of all the optimizers w.r.t their prediction accuracy and loss at both MSE and RMSE.
        In the next grid testing results of best performing combination will be plotted based on user inputs of loss and optimizer.
        ------------------------------------------------------------------------
        Parameters:
        targetFeature: Input dependent feature title; dtype -> str
        Acclim: y-limit input for accuracy; dtype -> float
        lslim: y-limit input for loss; dtype -> float
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_ANN_combination and saves the plot: plotOverall_results_lossinput.png
        ========================================================================
        '''
        self.targetFeature = targetFeature                                                  # Assigning target feature title
        self.store_coef_detMSE,self.store_MSE_ls = [],[]                                    # Store calculated loss and accuracy @ MSE
        self.store_coef_detRMSE,self.store_RMSE_ls = [],[]                                  # Store calculated loss and accuracy @ RMSE

        for index,loss in enumerate(self.loss): 
            fig = plt.figure(figsize=(11,5))                                                # Setting plot size
            axes = plt.subplot(121)                                                         # Setting subplots[1 row 2 columns]
            twin_axis = axes.twinx()                                                        # Creating twin axis for the loss

            axes.set_title('ANN Model for '+str(self.targetFeature)+' (@'+str(loss)+')')    # Set plot title
            axes.set_xlabel('Optimizers')                                                   # Set label for x-axis -> will be the optimizers
            axes.set_ylabel('$R^2$ value')                                                  # Set label for y-axis -> will be the prediction accuracy
            twin_axis.set_ylabel('Loss value')                                              # Set label for twin y-axis -> will be the prediction loss
            axes.set_ylim(Acclim[0],Acclim[1])                                              # Set y-lim for accuracy based on user inputs
            twin_axis.set_ylim(lslim[0],lslim[1])                                           # Set y-lim for loss based on user inputs 
            
            if loss== 'MSE':                                                                # Calculating loss and accuracy @ MSE
                for idx in range(len(self.optimizer)):
                    self.calculate_lossandRsqr(idx)                                         # Calculating loss and accuracy for all the optimizer @ MSE
                    self.store_coef_detMSE.append(self.Rsqr)                                # Storing the calculated accuracy
                    self.store_MSE_ls.append(self.loss_MSE)                                 # Storing the calculated loss
                    x,y = self.store_MSE_ls,self.store_coef_detMSE       
            else:
                twin_axis.set_ylim(lslim[2],lslim[3])
                for idx in range(len(self.optimizer)):                                      # Calculating loss and accuracy @ RMSE
                    self.calculate_lossandRsqr(idx+4)                                       # Calculating loss and accuracy for all the optimizer @ RMSE
                    self.store_coef_detRMSE.append(self.Rsqr)                               # Storing the calculated accuracy
                    self.store_RMSE_ls.append(self.loss_RMSE)                               # Storing the calculated loss
                    x,y = self.store_RMSE_ls,self.store_coef_detRMSE
            
            axes.bar(self.optimizer,y,edgecolor='C0',fill=False,width=0.4,label='$R^2$')    # Making bar chart of the calculated accuracy w.r.t the optimizers
            twin_axis.plot(x,'o--',color='C1',label=str(loss))                              # Making plot of the calculated loss in the twin axis w.r.t the optimizer

            plt1, label1 = axes.get_legend_handles_labels()                                 # Creating labels for axis and twin axis
            plt2, label2 = twin_axis.get_legend_handles_labels()
            twin_axis.legend(plt1 + plt2, label1 + label2, loc=0,ncol=2)

            # Making plot for the testing results of best performing combination
            axes1 = plt.subplot(122)
            if loss== 'MSE':                                                                # Loss based on user input loss
                index = self.optimizer.index(self.op)                                       # Select best performing optimizer among the optimizers 
                self.calculate_lossandRsqr(index)                                           # Calculating loss and accuracy for the user input optimizer 
                label = '($R^2$=%.3f; MSE=%.4f)'%(self.Rsqr,self.loss_MSE)                  # Creating labels
            else:
                index = self.optimizer.index(self.op)+4                                     # In case of RMSE loss as user input
                self.calculate_lossandRsqr(index)
                label = '($R^2$=%.3f; RMSE=%.4f)'%(self.Rsqr,self.loss_RMSE)
                
            self.rescaled_overall_testing = self.overall_testingResult*self.std_y + self.mean_y # Rescaling the results to its original metric for plotting
            axes1.set_title('Target vs Prediction (@'+str(self.op)+')')                         # Set plot title
            axes1.set_xlabel('No of samples')                                                   # Set label for x-axis -> will be the no of sample
            axes1.set_ylabel('Sample values')                                                   # Set label for y-axis -> will be the values of each sample
            axes1.plot(self.y_test*self.std_y + self.mean_y,'o--',markersize = 7,color = 'red', label = str(self.targetFeature))
            axes1.plot(self.rescaled_overall_testing[index],'o-',markersize = 4,color = 'blue', label = 'Predicted'+str(label))
            axes1.legend()
            fig.tight_layout()                                                                  # Set tight layout

            # Creating a directory if one does not exists
            #------------------------------------------------------------------------
            if not os.path.exists('Plots_ANN_combination'):
                os.makedirs('Plots_ANN_combination') 
            save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory

            fig = plt.gcf()                                                                     # Get the current figure
            fig.set_size_inches((12,5.7),forward = False)                                       # Assigning plot size (width,height)
            fig.set_dpi(300)                                                                    # Set resolution in dots per inch
            fig.savefig(os.path.join(save_path, 'plotOverall_results_'+str(loss)+'.png'),bbox_inches='tight') # Saving the plot
            plt.clf()                                                                           # Cleaning the current figure after saving
            
    def plot_OverallTestingResult(self):
        '''
        ========================================================================
        Description:
        Creating overall testing result for all the optimizer at MSE and RMSE loss.
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_ANN_combination and saves the plot: plot_Overall_TestingResult_lossinput.png
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Plots_ANN_combination'):
            os.makedirs('Plots_ANN_combination') 
        save_path = os.path.abspath('Plots_ANN_combination') # Save the file to the created directory

        # Plot Testing results for all combinations of loss and optimizer 
        #------------------------------------------------------------------------
        for ls in self.loss:
            for op_idx,op in enumerate(self.optimizer):

                idx = op_idx if ls == 'MSE' else op_idx + 4                                     # Accessing optimizer values based on loss inputs

                plt.suptitle('Overall Testing result: Target vs Prediction (@'+str(ls)+')')     # Creating suptitle for the plot
                plot_position = [(0,0),(0,1),(1,0),(1,1)]                                       # Creating grid positions for the plot
                plot_axis = ['ax1','ax2','ax3','ax4']                                           # Creating axis for the plot w.r.t the no of optimizers
                plot_axis[op_idx] = plt.subplot2grid((2,2),plot_position[op_idx])               # Plot in 2 x 2 grids
                self.calculate_lossandRsqr(idx)                                                 # Calculate loss and optimizer for each combination
                loss = self.loss_MSE if ls == 'MSE' else self.loss_RMSE                         # Assigning loss values
                ex_label = '($R^2$=%.3f; loss=%.4f)'%(self.Rsqr,loss)                           # Creating label for the plot
                plot_axis[op_idx].set_xlabel('No of samples')                                   # Set label for x-axis -> will be the no of sample 
                plot_axis[op_idx].set_ylabel('Sample values')                                   # Set label for y-axis -> will be the values of each sample
                plot_axis[op_idx].plot(self.y_test*self.std_y + self.mean_y,'o--',markersize = 7,color = 'red', label = str(self.targetFeature))
                plot_axis[op_idx].plot(self.rescaled_overall_testing[idx],'o-',markersize = 4,color = 'blue', label = 'Predicted@'+str(op)+str(ex_label))
                plot_axis[op_idx].legend(loc='lower left',prop={'size':7.8},ncol=2)             # Optimizing legends size and writing legends in two columns

            plt.tight_layout()                                                                  # Set tight layout
            fig = plt.gcf()                                                                     # Get the current figure
            fig.set_size_inches((11.5,7),forward = False)                                       # Assigning plot size (width,height)
            fig.set_dpi(100)                                                                    # Set resolution in dots per inch
            fig.savefig(os.path.join(save_path, 'plot_Overall_TestingResult_'+str(ls)+'.png'),bbox_inches='tight') # Saving the plot
            plt.clf()                                                                           # Cleaning the current figure after saving

def check_inputs(no_of_epoch,loss,optimizer,layers):
    '''
    ========================================================================
    Description:
    To check whether certain user inputs are within the required limits.
    ------------------------------------------------------------------------
    Parameters:
    no_of_epoch: Training size; dtype -> int
    loss: Loss input from best performing combinations; dtype -> str
    optimizer: Optimizer input from best performing combinations; dtype -> str
    layer: Hiddenlayers; dtype -> int 
    ------------------------------------------------------------------------
    Note:
    -If the inputs are not within the options range, program exits by throwing an error with possible inputs
    ========================================================================
    '''
    # Checking whether the input is correct or wrong
    #------------------------------------------------------------------------
    
    if (not loss == 'MSE') and (not loss == 'RMSE'):                         # Checking whether the give loss input is within the options range
        sys.exit('Error: Recheck loss input\nPossible inputs: MSE or RMSE')  # Else program exits by throwing an error mentioning possible inputs
    if(no_of_epoch <= 0):                                                    # Checking epoch input, it should be greater than zero
        sys.exit('Error: no_of_epoch input must be > 0')                     # Program exits if the input is lesser than or equal to zero
    if(layers[0]<=0 or layers[1]<=0):                                        # Checking hidden layer inputs, it should be greater than zero
        sys.exit('Error: layers input must be > 0')                          # Program exits if any one of the input is lesser than or equal to zero
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
@click.option('--loss',nargs=1,type=str,default='MSE',help='Select best performing loss: [MSE or RMSE] for HL')
@click.option('--optimizer',nargs=1,type=str,default='Adam',help='Select best performing optimizer: [SGD,SGDM,RMSP or Adam] for HL')
@click.option('--losslist',nargs=2,type=str,default=['MSE','RMSE'],help='Loss input as a list')
@click.option('--optimizerlist',nargs=4,type=str,default=['SGD','SGDM','RMSP','Adam'],help='Optimizer input as a list')
@click.option('--layers',nargs=2,type=int,default=([10,10]),help='Enter hidden layer(N1,N2) input for [IP-N1-N2-OP](>0) based on HP tuning')
@click.option('--sgd',nargs=2,type=float,default=([0.85,1e-1]),help='Enter SGD_optimizer input')
@click.option('--sgd_rmse',nargs=2,type=float,default=([0.85,1e-2]),help='Enter SGD_optimizer input @RMSE')
@click.option('--sgdm',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input')
@click.option('--sgdm_rmse',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input @RMSE')
@click.option('--rmsp',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input')
@click.option('--rmsp_rmse',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input @RMSE')
@click.option('--adam',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input')
@click.option('--adam_rmse',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input @RMSE')
@click.option('--hl1list',nargs=4,type=int,default=([70,50,20,10]),help='Enter hidden layer1 list(N1) for HP tuning')
@click.option('--hl2list',nargs=4,type=int,default=([70,50,20,10]),help='Enter hidden layer2 list(N2) for HP tuning')
@click.option('--selectdata',nargs=6,type=int,default=([500,50,500,50,500,20]),help='Selecting value range to generate readable plots for loss_Acc,lr & HL')
@click.option('--targetf',nargs=1,type=str,default='Fatigue Strength',help='Enter target feature')
@click.option('--limacc',nargs=2,type=float,default=([0.96,0.99]),help='Enter lw & up bound among Acc val')
@click.option('--limls',nargs=4,type=float,default=([0.02,0.032,0.114,0.126]),help='Enter lw & up bound @MSE & RMSE')
#
# ============================================================================================================================================
#                                                      CREATING COMBINATIONS WITH RESULTS --> ANN REGRESSION
# ============================================================================================================================================
#
def ANN_combination(data,no_of_epoch,loss,optimizer,losslist,optimizerlist,layers,sgd,sgd_rmse,sgdm,sgdm_rmse,rmsp,rmsp_rmse,adam,adam_rmse,\
    hl1list,hl2list,selectdata,targetf,limacc,limls):
    '''
    ========================================================================
    Description:
    This CREATING COMBINATIONS WITH RESULTS make comparison plots for all the training results and testing results, creates a comparison Excel for all 
    prediction results, make comparison plots for four different combination of hidden layers and also writes an comparison table for time taken by the 
    model to execute at different instances of the code.  
    ========================================================================
    '''
    # Check certain user inputs
    #-----------------------------------------------------------------------
    check_inputs(no_of_epoch,loss,optimizer,layers)

    # Adjustable hyperparameters must be tunned separately w.r.t the dataset used as it can impact the accuracy of the model
    # All the inputs must be same as that of PPP_ANN_createOverallresults.py 
    # Creating the input dictionary of hp
    #------------------------------------------------------------------------     
    N1,N2 = layers                                                  # Layer inputs in form of tuple
    SGD_lr,SGD_lrd = sgd                                            # SGD hyperparameter inputs in form of tuple @MSE
    SGD_lrR,SGD_lrdR = sgd_rmse                                     # SGD hyperparameter inputs in form of tuple @RMSE
    SGDM_lr,SGDM_lrd,SGDM_m = sgdm                                  # SGDM hyperparameter inputs in form of tuple @MSE
    SGDM_lrR,SGDM_lrdR,SGDM_mR = sgdm_rmse                          # SGDM hyperparameter inputs in form of tuple @RMSE
    RMSP_lr,RMSP_lrd,RMSP_ep,RMSP_rho = rmsp                        # RMSP hyperparameter inputs in form of tuple @MSE
    RMSP_lrR,RMSP_lrdR,RMSP_epR,RMSP_rhoR = rmsp_rmse               # RMSP hyperparameter inputs in form of tuple @RMSE
    Adam_lr,Adam_lrd,Adam_ep,Adam_b1,Adam_b2 = adam                 # Adam hyperparameter inputs in form of tuple @ MSE
    Adam_lrR,Adam_lrdR,Adam_epR,Adam_b1R,Adam_b2R = adam_rmse       # Adam hyperparameter inputs in form of tuple @ RMSE

    hyp_paramANN = {
        'layers':{'hidden_layer1':N1, 'hidden_layer2':N2, 'output_layer':1},
        'SGD':{'learning_R':SGD_lr, 'learning_R_decay':SGD_lrd},
        'SGD_RMSE':{'learning_R':SGD_lrR, 'learning_R_decay':SGD_lrdR},                                                         
        'SGDM':{'learning_R':SGDM_lr, 'learning_R_decay':SGDM_lrd, 'momentum':SGDM_m},
        'SGDM_RMSE':{'learning_R':SGDM_lrR, 'learning_R_decay':SGDM_lrdR, 'momentum':SGDM_mR},                                             
        'RMSP':{'learning_R':RMSP_lr, 'learning_R_decay':RMSP_lrd, 'epsilon':RMSP_ep, 'rho':RMSP_rho},
        'RMSP_RMSE':{'learning_R':RMSP_lrR, 'learning_R_decay':RMSP_lrdR, 'epsilon':RMSP_epR, 'rho':RMSP_rhoR},
        'Adam':{'learning_R':Adam_lr, 'learning_R_decay':Adam_lrd, 'epsilon':Adam_ep, 'beta1':Adam_b1, 'beta2':Adam_b2},
        'Adam_RMSE':{'learning_R':Adam_lrR, 'learning_R_decay':Adam_lrdR, 'epsilon':Adam_epR, 'beta1':Adam_b1R, 'beta2':Adam_b2R} } 
    
    listlayers = []                                                 # Store different hidden layer combinations
    for idx in range (len(hl1list)):
        listlayers.append([hl1list[idx],hl2list[idx]])              # Creating hidden layer combinations

    # Initializing results comparison
    #------------------------------------------------------------------------
    results_comparision = Result_comparison(data,no_of_epoch,loss ,optimizer,losslist ,optimizerlist,hyp_paramANN,listlayers,selectdata)

    results_comparision.read_data()                                 # Read data and store results

    results_comparision.write_TargetVSpred()                        # Write comparison Excel of prediciton results
    results_comparision.write_time()                                # Write tabulated time taken

    results_comparision.plot_OverallLoss()                          # Create overall loss plot @ MSE & RMSE
    results_comparision.plot_OverallAccuracy()                      # Create overall accuracy plot @ MSE & RMSE
    results_comparision.plot_OveralllearningR()                     # Create learning rate relation plot with loss @ MSE & RMSE
    results_comparision.plot_hypHiddenlayer()                       # Create hidden layer combination plot @ best combination loss and optimizer

    results_comparision.plot_OverallResult(targetf,limacc,limls)    # Create overall result plots for analysis @ MSE & RMSE
    results_comparision.write_Modelperformance()                    # Write results of all combinations during training and prediction
    results_comparision.plot_OverallTestingResult()                 # Create overall testing results @ MSE & RMSE
    
if __name__ == '__main__':
   ANN_combination()