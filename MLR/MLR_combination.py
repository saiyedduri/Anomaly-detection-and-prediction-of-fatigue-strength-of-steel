'''
========================================================================
This program is a part of personal programming project
MLR program No: 3 
file: PPP_MLR_combination
------------------------------------------------------------------------
Creates a model analysis plot by reading in all the results created for loss combinations
Creates an comparison Excel of target and the predicted values for loss combinations and SGD optimizer
Creates a comparison table for the time taken by the model to perform a particular task at different instances for loss combinations
These plots and datas are then used to analyse the model performance
Condition1: This program should be executed only after executing PPP_MLR_createOverallresults.py @ predictor: OFF
Condition2: The input values should be same as the values that was used to create these datas by executing PPP_MLR_createOverallresults.py
'''
# Import created libraries
# -------------------------
from PPP_MLR import Data_preprocessing

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
# Creating plots and datas for all the combinations of MLR Regression model 
# ======================================================================
#
class Result_comparison:
    '''
    ========================================================================
    Description: 
    Creating model performance analysis plots for the results attained from loss combinations of loss and SGD optimizer in MLR regression.
    Making comparison Excels for prediciton results and tabulating the time results for loss combinations.
    ========================================================================
    '''

    def __init__(self,data,ls,losslist):
        '''
        ========================================================================
        Description:
        Initializing the inputs and created libraries, performing data preprocessing steps and 
        creating storage to store data.  
        ------------------------------------------------------------------------
        Parameters:
        data: Input dataset in .csv format with dependent feature as the last column
        ls: Loss input from the best performing combination and the same is used to create hidden layer combinations; dtype -> str
        losslist: Loss input list; dtype -> str
        ========================================================================
        '''
        self.ls = ls
        self.loss = losslist

        # Data_preprocessing
        # ------------------------------------------------------------------------
        Preprocessed_data = Data_preprocessing()                     # Initialization for data preprocessing
        X,y = Preprocessed_data.import_dataset(data)                 # X: Independent feature, y: dependent feature
        scaled_X,scaled_y,self.mean_y,self.std_y = Preprocessed_data.feature_scaling(X,y) # Scaled features, mean and std for rescaling during comparison
        self.X_train, self.X_test,self.y_train,self.y_test = Preprocessed_data.split_dataset(scaled_X,scaled_y) # Splitting dataset into training and test set

        # Creating Storage to store read in data of loss combination to make plots and data comparison
        #------------------------------------------------------------------------
        self.pred_loss,self.pred_Acc = [],[]                                    # To store prediction loss and accuracy
        self.pred_MSE,self.pred_RMSE = [],[]                                    # To store prediction results @MSE & RMSE
        self.timefw,self.timebw,self.timeop = [],[],[]                          # To store overall time taken by forward prop, backward prop and optimization to execute during training
        self.timemod,self.timetrain,self.timepred = [],[],[]                    # To store overall time taken by model, training and testing to execute
        self.timecode = []                                                      # To store overall time taken by each code to execute

    def read_data(self):
        '''
        ========================================================================
        Description:
        To read in data from loss combinations to make comparison plots.
        Reading in loss and accuracy values at each loss from the file: testingresults_lossinput.txt in the directory: Results_Testing.
        Reading in predicted results at each loss from the file: resultcomparision_lossinput.txt in the directory: Results_TargetVSpred.
        Reading in the time results from the file1: resultcomparison_lossinput.txt and file2: timetaken_overall.txt in the directory: Results_TargetVSpred.
        ========================================================================
        '''
        # Read loss and accuracy from each loss combination
        #------------------------------------------------------------------------ 
        for ls_ip in self.loss:
            # Creating paths and filenames to open
            #------------------------------------------------------------------------
            open_path = os.path.abspath('Results_Testing')                           # Creating path1 to open directory -> Results_Testing
            open_path1 = os.path.abspath('Results_TargetVSpred')                     # Creating path2 to open directory -> Results_TargetVSpred
            filename = os.path.join(open_path,'testingresults_%s.txt' % (ls_ip))     # File1 to open -> testingresults_lossinput.txt
            filename1 = os.path.join(open_path1,'timetaken_overall.txt')             # File2 to open -> timetaken_overall.txt

            # Reading results from testingresults_lossinput.txt
            #------------------------------------------------------------------------
            with open(filename, 'r') as f:
                read_data = f.readlines()
                data = read_data[4].split(':')
                self.pred_loss.append(float(data[3].split('\n')[0]))                # Reading in the loss value of each loss combination
                self.pred_Acc.append(float(data[2].split(',')[0]))                  # Reading in the accuracy value of each loss combination

        # Read predicted results and time values from Results_TargetVSpred
        #------------------------------------------------------------------------
        for ls_ip in self.loss:
            filename2 = os.path.join(open_path1,'resultcomparison_%s.txt' % (ls_ip)) # File3 to open -> resultcomparison_lossinput.txt
            with open(filename2, 'r') as f:
                read_data = f.readlines()
                for idx in (np.arange(7,2*len(self.y_test)+7,2)):
                    if ls_ip == 'MSE':
                        self.pred_MSE.append(float(read_data[idx].split('|')[2]))   # Reading in prediction results @MSE loss 
                    else:
                        self.pred_RMSE.append(float(read_data[idx].split('|')[2]))  # Reading in prediction results @RMSE loss

                # Read time to make time comparision table
                #------------------------------------------------------------------------
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

        # Reading results from timetaken_overall.txt
        #------------------------------------------------------------------------
        with open(filename1, 'r') as f:
            read_data = f.readlines()                                               # Reading time taken results
            for idx in (np.arange(7,2*len(self.loss)+7,2)):
                self.timecode.append(float(read_data[idx].split('|')[2]))           # Storing time taken by each code to execute in PPP_MLR_createOverallresults.py
            self.timeall = float(read_data[2*len(self.loss)+8].split(':')[1])       # Time taken by loss combination to execute in PPP_MLR_createOverallresults.py
    
    def write_TargetVSpred(self):
        '''
        ========================================================================
        Description:
        To create a comparison Excel between the target results and the predicted results for the loss combinations and SGD optimizer. 
        From this Excel we can compare the predicted results of all the combinations for each sample(line item) against the target results and perform result analysis.
        ------------------------------------------------------------------------
        Conditions:
        Make sure xlsxwriter is available or pip install xlsxwriter
        ------------------------------------------------------------------------
        Output:
        Writes a file: resultcomparisonMLR.xlsx to the directory: Results_TargetVSpred         
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_TargetVSpred'):
            os.makedirs('Results_TargetVSpred')
        save_path = os.path.abspath('Results_TargetVSpred') # Save the file to the created directory

        # Creating inputs for each column of the dataframe from the stored prediction results
        #------------------------------------------------------------------------
        self.target = (self.y_test*self.std_y + self.mean_y).reshape(-1,)       # Rescaling target feature
        c1,c2,c3 = self.target, self.pred_MSE, self.pred_RMSE

        # Creating a dataframe to write results in a tablulated xlsx format
        #------------------------------------------------------------------------
        writedata = pd.DataFrame({'Target': c1,'Pred_MSE': c2,'Pred_RMSE': c3})
        
        # Writing an Excel
        #------------------------------------------------------------------------
        writer = pd.ExcelWriter(os.path.join(save_path, 'resultcomparisonMLR.xlsx'))                          # Creating an Excel writer
        writedata.to_excel(writer, sheet_name='targetVSpred',float_format = '%.0f',index = False,startrow=1)  # Converting dataframe into an Excel by specifying the sheet name,
                                                                                                              # floating format, starting row and avoiding dataframe index
        # Getting xlsxwriter objects from the dataframe writer object
        #------------------------------------------------------------------------
        getworkbook = writer.book
        getworksheet = writer.sheets['targetVSpred']                                                          # Specify the sheet name

        writetitle = 'Result comparison: Target vs Predicted(MLR)'                                            # Setting title for the Excel
        getworksheet.write(0,0,writetitle)                                                                    # Writing the title in the Excel at a specified location
        cellformat_title = getworkbook.add_format({'bold':True,'font_size': 14})                              # Formating the title by making it bold and setting a font size
        getworksheet.set_row(0,40,cellformat_title)                                                           # Applying the formating and setting the title row height

        cellformat_border = getworkbook.add_format({'border':1})                                              # Setting closed borders to the Excel
        getworksheet.conditional_format('A1:C%d'%(len(self.y_test)+2), {'type':'no_blanks','format':cellformat_border}) # Applying boarder settings for required rows and cols
        cellformat_column = getworkbook.add_format({'align':'center'})                                        # Setting center alignment to the datas
        getworksheet.set_column('A1:C%d'%(len(self.y_test)+2), 16,cellformat_column)                          # Applying the alignment setting and adjusting the column length
        
        # Using dataframe to calculate the mean and the std of each combination and writing it in the Excel
        #------------------------------------------------------------------------
        mean,std = np.array(writedata.mean()),np.array(writedata.std())                                       # Calculating mean and std
        writemean = pd.DataFrame({'Target': [mean[0]],'Pred_MSE': [mean[1]],'Pred_RMSE': [mean[2]]})          # Creating a dataframe of mean
        writestd = pd.DataFrame({'Target': [std[0]],'Pred_MSE': [std[1]],'Pred_RMSE': [std[2]]})              # Creating a dataframe of std
        writemean.to_excel(writer, sheet_name='targetVSpred',float_format = '%.3f',header = False,index = False,startrow=len(self.y_test)+3) # Converting dataframe into an 
        writestd.to_excel(writer, sheet_name='targetVSpred',float_format = '%.3f',header = False,index = False,startrow=len(self.y_test)+4)  # Excel in both cases

        bold = getworkbook.add_format({'bold': 1})                                                            # Setting bold formating
        getworksheet.write(len(self.y_test)+3,3,'Mean',bold)                                                  # Writing the mean in the Excel at a specified location
        getworksheet.write(len(self.y_test)+4,3,'Std',bold)                                                   # Writing the std in the Excel at a specified location

        # Adding Notes
        #------------------------------------------------------------------------
        note ='Note: Pred_x is predicted values using x: loss'                                                # Setting additional note
        getworksheet.write(len(self.y_test)+6,0,note)                                                         # Writing the note in the Excel at a specified location
        cellformat_note = getworkbook.add_format({'bold':True,'font_size': 14})                               # Formating the note by making it bold and setting a font size
        getworksheet.set_row(len(self.y_test)+6,20,cellformat_note)                                           # Applying the formating and setting the title row height
        writer.save()                                                                                         # Close the Excel writer and output the Excel file

    def write_time(self):
        '''
        ========================================================================
        Description:
        To create a comparision table that compares the time taken by the model at different instances for all loss combinations. 
        ------------------------------------------------------------------------
        Conditions:
        Make sure tabulate package is installed or pip install tabulate
        ------------------------------------------------------------------------
        Output:
        Creates a directory: Results_Timetaken and and writes a file: resulttimetakenMLR.txt
        ========================================================================
        '''
        # Creating a directory if one does not exists
        #------------------------------------------------------------------------
        if not os.path.exists('Results_Timetaken'):
            os.makedirs('Results_Timetaken')
        save_path = os.path.abspath('Results_Timetaken') # Save the file to the created directory

        # Creating a dataframe for time taken results so that the results can be written in a tablulated form
        #------------------------------------------------------------------------
        comb = ['MSE_SGD','RMSE_SGD'] # Combinations list to tabulate
        writedata = pd.DataFrame({'Combinations': comb,'Code':self.timecode,'Model':self.timemod,'Training': self.timetrain,'Prediction': self.timepred,\
            'Forward_prop': self.timefw,'Backward_prop': self.timebw,'Optimizer': self.timeop})

        # writing time taken results into the file -> resulttimetakenMLR.txt
        #------------------------------------------------------------------------
        with open(os.path.join(save_path, 'resulttimetakenMLR.txt'), 'w') as f:

            print('Time taken in seconds by MLR at different instances and combinations:\n',file=f) # Creating title infos before writing 
            print(writedata.to_markdown(tablefmt='grid',floatfmt='.6f',index=False),file=f)         # Tabulating the results in grid format without index
            print('\nTime taken by two combination to execute:',self.timeall,file=f)                # Writing results of time taken for two combinations

    def plot_OverallResult(self,targetFeature,Acclim,lslim):
        '''
        ========================================================================
        Description:
        Creating an analysis plot comparing the performance of all the losses w.r.t their prediction accuracy and loss.
        In the next grid testing results of best performing combination will be plotted based on user inputs of loss and optimizer.
        ------------------------------------------------------------------------
        Parameters:
        targetFeature: Input dependent feature title; dtype -> str
        Acclim: y-limit input for accuracy; dtype -> float
        lslim: y-limit input for loss; dtype -> float
        ------------------------------------------------------------------------
        Output:
        Creates directory if one does not exist: Plots_MLR_combination and saves the plot: plotOverall_results.png
        ========================================================================
        '''
        fig = plt.figure(figsize=(11,5))                                                         # Setting plot size                                                                                                            
        axes = plt.subplot(121)                                                                  # Setting subplots[1 row 2 columns]
        twin_axis = axes.twinx()                                                                 # Creating twin for the loss

        axes.set_title('MLR Model for '+str(targetFeature)+' (@SGD)')                            # Set plot title
        axes.set_xlabel('Losses')                                                                # Set label for x-axis -> will be the losses
        axes.set_ylabel('$R^2$ value')                                                           # Set label for y-axis -> will be the prediction accuracy
        twin_axis.set_ylabel('Loss value')                                                       # Set label for twin y-axis -> will be the prediction loss
        axes.set_ylim(Acclim[0],Acclim[1])                                                       # Set y-lim for accuracy based on user inputs
        twin_axis.set_ylim(lslim[0],lslim[1])                                                    # Set y-lim for loss based on user inputs 
        axes.bar(self.loss, self.pred_Acc,edgecolor='C0',fill=False,width=0.4,label='$R^2$')     # Making bar chart of the calculated accuracy w.r.t the losses
        twin_axis.plot(self.pred_loss,'o--',color='C1',label='loss')                             # Making plot of the calculated loss in the twin axis w.r.t the losses

        plt1, label1 = axes.get_legend_handles_labels()                                          # Creating labels for axis and twin axis
        plt2, label2 = twin_axis.get_legend_handles_labels()
        twin_axis.legend(plt1 + plt2, label1 + label2, loc=0,ncol=2)

        # Making plot for the testing results of best performing combination
        axes1 = plt.subplot(122)
        loss = 'MSE' if self.ls == 'MSE' else 'RMSE'                                             # Loss based on user input loss
        idx = 0 if self.ls == 'MSE' else 1                                                       # Selecting loss and accuracy values based on user input loss
        plot_ls = self.pred_MSE if self.ls == 'MSE' else self.pred_RMSE                          # Selecting prediction results based on user input loss

        label = '($R^2$=%.3f; %s=%.4f)'%(self.pred_Acc[idx],loss,self.pred_loss[idx])            # Creating labels for the plot
        axes1.set_title('Target vs Prediction (@SGD)')                                           # Set plot title
        axes1.set_xlabel('No of samples')                                                        # Set label for x-axis -> will be the no of sample
        axes1.set_ylabel('Sample values')                                                        # Set label for y-axis -> will be the values of each sample
        axes1.plot(self.target, 'o--',color = 'red', markersize = 7, label = str(targetFeature))
        axes1.plot(plot_ls,'o-',color = 'blue',markersize = 4, label = 'Predicted'+str(label))
        axes1.legend()
        fig.tight_layout()                                                                       # Set tight layout

        # Creating a directory if one does not exists
        #------------------------------------------------------------------------                                                               
        if not os.path.exists('Plots_MLR_combination'):
            os.makedirs('Plots_MLR_combination') 
        save_path = os.path.abspath('Plots_MLR_combination') # Save the file to the created directory

        fig = plt.gcf()                                                                          # Get the current figure
        fig.set_size_inches((12,5.7),forward = False)                                            # Assigning plot size (width,height)
        fig.set_dpi(300)                                                                         # Set resolution in dots per inch
        fig.savefig(os.path.join(save_path, 'plotOverall_results.png'),bbox_inches='tight')      # Saving the plot
        plt.clf()                                                                                # Cleaning the current figure after saving
#
# ======================================================================
# User selection of Adjustable inputs -> Implementing click
# ======================================================================
#
@click.command()
@click.option('--data',nargs=1,type=str,default='fatigue_dataset.csv',help='Enter input dataset.csv: last column must be the target feature')
@click.option('--loss',nargs=1,type=str,default='MSE',help='Select best performing loss: [MSE or RMSE] for overallplot')
@click.option('--losslist',nargs=2,type=str,default=['MSE','RMSE'],help='Loss input as a list')
@click.option('--targetf',nargs=1,type=str,default='Fatigue Strength',help='Enter target feature')
@click.option('--limacc',nargs=2,type=float,default=([0.95,0.97]),help='Enter lw & up bound among Acc val')
@click.option('--limls',nargs=2,type=float,default=([0.02,0.15]),help='Enter lw & up bound @MSE & RMSE')
# ============================================================================================================================================
#                                                      CREATING COMBINATIONS WITH RESULTS --> MLR REGRESSION
# ============================================================================================================================================
def MLR_combination(data,loss,losslist,targetf,limacc,limls):
    '''
    ========================================================================
    Description:
    This CREATING COMBINATIONS WITH RESULTS make overall result plots for analysis, creates a comparison Excel for all 
    prediction results and writes an comparison table for time taken by the model to execute at different instances of the code.  
    ========================================================================
    '''
    # Checking whether the input is correct or wrong
    #------------------------------------------------------------------------
    if (not loss == 'MSE') and (not loss == 'RMSE'):                        # Checking whether the give loss input is within the options range
        sys.exit('Error: Recheck loss input\nPossible inputs: MSE or RMSE') # If not program exits by throwing an error mentioning possible inputs

    # Initializing results comparision
    #------------------------------------------------------------------------
    results_comparison = Result_comparison(data,loss,losslist)
    results_comparison.read_data()                                         # Read data and store results

    results_comparison.write_TargetVSpred()                                # Write comparision Excel of prediciton results
    results_comparison.write_time()                                        # Write tabulated time taken
    
    results_comparison.plot_OverallResult(targetf,limacc,limls)            # Create overall result plots for analysis

if __name__ == '__main__':
   MLR_combination()