'''
========================================================================
This program is a part of personal programming project
ANN program No: 2 
file: PPP_ANN_createOverallresults
------------------------------------------------------------------------
Creates results for all the eight combinations of loss and optimizer with Predictor ON and OFF in a single attempt
Used to tune all the hyperparameters especially optimizers of the implemented model in a single go
Creates model loss and testing results for four different combinations of hidden layer inputs to analyse the hidden layer hyperparameter
The created results are then used in file: PPP_ANN_combination.py to make plots with all the combinations for analysis
Condition: If the results are created with predictor: ON then only predictor results are created which is not sufficient to execute PPP_ANN_combination.py
======================================================================== 
'''
# Import required libraries
# -------------------------
import pandas as pd
import subprocess
import sys
import os
import click
import timeit
#
# ======================================================================
# Creating results for all the combinations of ANN Regression model 
# ======================================================================
#
class Create_Overallresults:
    '''
    ========================================================================
    Description: 
    Creating overall results for all the eight combinations of loss and optimizer in ANN regression.
    ========================================================================
    '''

    def createRes_toplot(self,data,no_of_epoch,pred_dataset,loss,optimizer,layers,reg,sgd,sgd_rmse,sgdm,sgdm_rmse,rmsp,rmsp_rmse,adam,adam_rmse,makeplot,writeparam_as,pred):
        '''
        ========================================================================
        Description:
        To create a master code that runs PPP_ANN_main.py using subprocess for all the combinations and for any different hyperparameters.
        This helps us to study the impact of tunning or changing any parameter on all the combinations.
        ------------------------------------------------------------------------
        Parameters:
        All parameter details including datatype and number of arguments required can be found in click options implemented below
        Execute the program PPP_ANN_createOverallresults.py --help to get more details
        ------------------------------------------------------------------------
        Return:
        timetaken_percode: Time taken per combination code to execute in seconds; dtype -> float    
        ========================================================================
        '''
        timetaken_percode = []  # Store time taken per combination code to execute

        # Using all the inputs to execute PPP_ANN_main.py w.r.t loss and optimizer
        #------------------------------------------------------------------------                                    
        for ls in loss: 
            for op in optimizer:
                # Selecting different hyperparameters in case of RMSE loss for each optimizers 
                if (ls == 'RMSE') and (op == 'SGD'):
                    sgd = sgd_rmse
                elif (ls == 'RMSE') and (op == 'SGDM'):
                    sgdm = sgdm_rmse
                elif (ls == 'RMSE') and (op == 'RMSP'):
                    rmsp = rmsp_rmse
                elif (ls == 'RMSE') and (op == 'Adam'):
                    adam = adam_rmse

                start_time = timeit.default_timer()                              # Start timer
                subprocess.call([sys.executable,'.\PPP_ANN_main.py', '--data=' '%s' % data, '--no_of_epoch=' '%d' % no_of_epoch, '--loss=' '%s' % ls, \
                '--optimizer=' '%s' % op, '--pred_dataset=' '%s' % pred_dataset, '--layers=' '%d' % layers[0], '%d' % layers[1], \
                '--reg=' '%s' % str(reg[0]), '%s' % str(reg[1]), '%s' % str(reg[2]), '%s' % str(reg[3]), '%s' % str(reg[4]), '%s' % str(reg[5]), \
                '--sgd=' '%s' % str(sgd[0]), '%s' % str(sgd[1]), '--sgdm=' '%s' % str(sgdm[0]), '%s' % str(sgdm[1]), '%s' % str(sgdm[2]), \
                '--rmsp=' '%s' % str(rmsp[0]), '%s' % str(rmsp[1]), '%s' % str(rmsp[2]), '%s' % str(rmsp[3]), \
                '--adam=' '%s' % str(adam[0]), '%s' % str(adam[1]), '%s' % str(adam[2]), '%s' % str(adam[3]), '%s' % str(adam[4]), \
                '--writeparam_as=' '%s' % writeparam_as, '--makeplot=' '%d' % makeplot, '--predictor=' '%s' % pred])
                timetaken_percode.append(timeit.default_timer()-start_time)      # Stop timer. The taken taken for each combination code to execute is noted

        return timetaken_percode
#
# ======================================================================
# Creating results for different hidden layer inputs in ANN Regression model 
# ======================================================================
#
class Hyptuning_HL:
    '''
    ========================================================================
    Description: 
    Creating results for all the four hidden layer combinations w.r.t a single loss and optimizer for ANN regression.
    ========================================================================
    '''    

    def hiddenlayer_tuning(self,data,no_of_epoch,pred_dataset,loss,optimizer,reg,sgd,sgdm,rmsp,adam,makeplot,writeparam_as,writehl,HL1list,HL2list):
        '''
        ========================================================================
        Description:
        To create results for four different hidden layer combinations with any different hyperparameters and recording the time taken by each HL combination to execute.
        This runs for a single combination of loss and optimizer at a time and writes four different results to the directory -> Results_forPlotting.
        ------------------------------------------------------------------------
        Parameters:
        All parameter details including datatype and number of arguments required can be found in click options implemented below
        Execute the program PPP_ANN_createOverallresults.py --help to get more details
        ------------------------------------------------------------------------
        Note:
        The reason to execute this block for a single combination of loss and optimizer is to tune the HL hyperparameter first with four different combinations and then find
        the best working one and apply it in Create_Overallresults and check the impact it has made on other models. 
        ========================================================================
        ''' 
        timetaken = []                                      # Store time taken for different hidden layer inputs to execute
        listlayers = []                                     # Store different hidden layer combinations
        for idx in range (len(HL1list)):
            listlayers.append([HL1list[idx],HL2list[idx]])  # Creating hidden layer combinations
      
        for hl in listlayers:
            start_time = timeit.default_timer()                  # Start timer
            subprocess.call([sys.executable,'.\PPP_ANN_main.py', '--data=' '%s' % data, '--no_of_epoch=' '%d' % no_of_epoch, '--loss=' '%s' % loss, \
            '--optimizer=' '%s' % optimizer, '--pred_dataset=' '%s' % pred_dataset, '--layers=' '%d' % hl[0], '%d' % hl[1], \
            '--reg=' '%s' % str(reg[0]), '%s' % str(reg[1]), '%s' % str(reg[2]), '%s' % str(reg[3]), '%s' % str(reg[4]), '%s' % str(reg[5]), \
            '--sgd=' '%s' % str(sgd[0]), '%s' % str(sgd[1]), '--sgdm=' '%s' % str(sgdm[0]), '%s' % str(sgdm[1]), '%s' % str(sgdm[2]), \
            '--rmsp=' '%s' % str(rmsp[0]), '%s' % str(rmsp[1]), '%s' % str(rmsp[2]), '%s' % str(rmsp[3]), \
            '--adam=' '%s' % str(adam[0]), '%s' % str(adam[1]), '%s' % str(adam[2]), '%s' % str(adam[3]), '%s' % str(adam[4]), \
            '--writeparam_as=' '%s' % writeparam_as, '--writehl=' '%d' % writehl, '--makeplot=' '%d' % makeplot])
            timetaken.append(timeit.default_timer()-start_time) # Stop timer. The taken taken by each hidden layer combination to execute is noted
        
        # Write results of time taken by each hidden layer combination to execute -> lossatHiddLayHL1HL2OL3_lossinput_optimizerinput.txt
        #------------------------------------------------------------------------
        open_path = os.path.abspath('Results_forPlotting')       # directory to open
        for idx,hl in enumerate (listlayers):
            with open(os.path.join(open_path,'lossatHiddLay%d%d%d_%s_%s.txt' % (hl[0],hl[1],1,loss,optimizer)) , 'a') as f: # Open file to write
        
                print('\nTime taken for IP-%d-%d-%d is %.3f seconds'%(hl[0],hl[1],1,timetaken[idx]),file=f) # Writing time taken results in seconds

def write_time(percode,overall):
    '''
    ========================================================================
    Description:
    To write results of time taken by all the eight combinations to execute and time taken by each combination to execute
    ------------------------------------------------------------------------
    Parameters:
    percode: time taken for each combination in seconds; dtype -> float
    overall: time taken for all combinations in seconds; dtype -> float
    ------------------------------------------------------------------------
    Conditions:
    Make sure tabulate package is installed or pip install tabulate
    ------------------------------------------------------------------------
    Output:
    writes a file: timetaken_overall.txt to the directory: Results_TargetVSpred     
    ========================================================================
    ''' 
    # Creating a dataframe for time taken results so that the results can be written in a tablulated form
    #------------------------------------------------------------------------
    comb = ['MSE_SGD','MSE_SGDM','MSE_RMSP','MSE_Adam','RMSE_SGD','RMSE_SGDM','RMSE_RMSP','RMSE_Adam'] # All the eight combinations list to tabulate
    writedata = pd.DataFrame({'Combinations':comb,'Time(s)':percode})

    # writing time taken results into the file -> timetaken_overall.txt
    #------------------------------------------------------------------------
    save_path = os.path.abspath('Results_TargetVSpred') # Save the file to the created directory
    with open(os.path.join(save_path,'timetaken_overall.txt') , 'w') as f:
        
        print('Time taken in create overall results',file=f)                                # Creating title infos before writing
        print('\nTime taken by each code to execute:\n', file=f)                            # Writing results of time taken for each combination
        print(writedata.to_markdown(tablefmt='grid',floatfmt='.6f',index=False),file=f)     # Tabulating the results in grid format without index
        print('\nTime taken by all the eight combination to execute:',overall,file=f)       # Writing results of time taken for all the combinations

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
    predictor: Predictor input whether to execute the model as a pure predictor; dtype -> str
    ------------------------------------------------------------------------
    Note:
    -If the inputs are not within the options range, program exits by throwing an error with possible inputs
    ========================================================================
    '''
    # Checking whether the input is correct or wrong
    #------------------------------------------------------------------------
    inputset1 = [pred_dataset,loss,makeplot,writeparam_as,writehl,predictor] # Grouping similar inputs and their options togather
    optionset1 = [['testset','trainingset','pred_dataset'],['MSE','RMSE','loss'],[0,1,'makeplot'],['npy','txt','writeparam_as'],[0,1,'writehl'],['ON','OFF','predictor']]

    for idx,input in enumerate(inputset1): # Checking for correctness
        if (not input == optionset1[idx][0]) and (not input == optionset1[idx][1]): # If the inputs are not within the options range program exits by throwing an error mentioning possible inputs
            sys.exit('Error: Recheck '+str(optionset1[idx][2])+' input\nPossible inputs: '+str(optionset1[idx][0]) +' or '+str(optionset1[idx][1]))
    
    if(no_of_epoch <= 0):                                  # Checking epoch input, it should be greater than zero
        sys.exit('Error: no_of_epoch input must be > 0')   # Program exits if the input is lesser than or equal to zero
    if(layers[0]<=0 or layers[1]<=0):                      # Checking hidden layer inputs, it should be greater than zero
        sys.exit('Error: layers input must be > 0')        # Program exits if any one of the input is lesser than or equal to zero
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
@click.option('--loss',nargs=1,type=str,default='MSE',help='Select loss: [MSE or RMSE] for HL')
@click.option('--optimizer',nargs=1,type=str,default='Adam',help='Select optimizer: [SGD,SGDM,RMSP or Adam] for HL')
@click.option('--losslist',nargs=2,type=str,default=['MSE','RMSE'],help='Loss input as a list')
@click.option('--optimizerlist',nargs=4,type=str,default=['SGD','SGDM','RMSP','Adam'],help='Optimizer input as a list')
@click.option('--layers',nargs=2,type=int,default=([10,10]),help='Enter hidden layer(N1,N2) input for [IP-N1-N2-OP](>0) based on HP tuning')
@click.option('--reg',nargs=6,type=float,default=([2e-6,2e-6,3e-6,3e-6,0,0]),help='Enter regularization loss(l2) for the layers')
@click.option('--sgd',nargs=2,type=float,default=([0.85,1e-1]),help='Enter SGD_optimizer input')
@click.option('--sgd_rmse',nargs=2,type=float,default=([0.85,1e-2]),help='Enter SGD_optimizer input @RMSE')
@click.option('--sgdm',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input')
@click.option('--sgdm_rmse',nargs=3,type=float,default=([0.85,1e-1,0.6]),help='Enter SGDM_optimizer input @RMSE')
@click.option('--rmsp',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input')
@click.option('--rmsp_rmse',nargs=4,type=float,default=([1e-3,1e-4,1e-7,0.9]),help='Enter RMSP_optimizer input @RMSE')
@click.option('--adam',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input')
@click.option('--adam_rmse',nargs=5,type=float,default=([1e-3,1e-5,1e-7,0.9,0.999]),help='Enter Adam_optimizer input @RMSE')
@click.option('--makeplot',nargs=1,type=int,default=0,help='Select 0 or 1 to makeplot')
@click.option('--writeparam_as',nargs=1,type=str,default='npy',help='select write format: npy or txt')
@click.option('--writehl',nargs=1,type=int,default=1,help='Select 0 or 1 to write hl results')
@click.option('--hl1list',nargs=4,type=int,default=([70,50,20,10]),help='Enter hidden layer1 list(N1) for HP tuning')
@click.option('--hl2list',nargs=4,type=int,default=([70,50,20,10]),help='Enter hidden layer2 list(N2) for HP tuning')
@click.option('--predictor',nargs=1,type=str,default='OFF',help='Select ON or OFF for only prediction')
#
# ============================================================================================================================================
#                                                      CREATING OVERALL RESULTS --> ANN REGRESSION
# ============================================================================================================================================
#
def ANN_createOverallresults(data,no_of_epoch,pred_dataset,loss,optimizer,layers,reg,sgd,sgd_rmse,sgdm,sgdm_rmse,rmsp,rmsp_rmse,adam,adam_rmse,makeplot,\
    writeparam_as,writehl,hl1list,hl2list,losslist,optimizerlist,predictor):
    '''
    ========================================================================
    Description:
    This CREATE OVERALL RESULTS, make prediction results for all the combination of losses and optimizers, tunes hyperparameters, and finds the resultant
    loss and prediction output for four different combination of hidden layers. In addition, it also notes the time taken by each combination to execute.
    ========================================================================
    '''
    # Check certain user inputs
    #-----------------------------------------------------------------------
    check_inputs(no_of_epoch,pred_dataset,loss,optimizer,layers,makeplot,writeparam_as,writehl,predictor)

    # Initializing overall results
    #------------------------------------------------------------------------
    createRes = Create_Overallresults()
    hyp_hiddenlayer = Hyptuning_HL()

    # Create Overall results for the combination to make plot and later analyse
    # Note: 
    # If predictor is ON all results of prediction for different combinations are present in the directory -> Results_ANN_predictor
    # The files present in the directory -> testingresults_lossinput_optimizerinput.txt and resultcomparision_lossinput_optimizerinput.txt
    #------------------------------------------------------------------------
    start_time = timeit.default_timer()                                      # Start timer to check the time taken for overall results
    timetaken_percode = createRes.createRes_toplot(data,no_of_epoch,pred_dataset,losslist,optimizerlist,layers,reg,sgd,sgd_rmse,sgdm,sgdm_rmse,rmsp,rmsp_rmse,adam,adam_rmse,makeplot,\
        writeparam_as,predictor)
    timetaken_overall = timeit.default_timer()-start_time                    # Stop time. The time taken to execute overall results is noted; dtype -> float
    
    # Write time taken result
    #------------------------------------------------------------------------
    write_time(timetaken_percode,timetaken_overall)

    # Hyperparameter tuning of Hiddenlayers and create results for different combinations to make plot and later analyse
    #------------------------------------------------------------------------
    if predictor == 'OFF':                                                   # If prediction is ON plotting hyperparameter tuning of Hiddenlayers will not execute
        hyp_hiddenlayer.hiddenlayer_tuning(data,no_of_epoch,pred_dataset,loss,optimizer,reg,sgd,sgdm,rmsp,adam,makeplot,writeparam_as,writehl,hl1list,hl2list)

if __name__ == '__main__':
   ANN_createOverallresults()
