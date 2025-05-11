DIRECTORY DETAILS
-----------------
Executable Files: 3
- PPP_ANN_main.py
- PPP_ANN_createOverallresults.py
- PPP_ANN_combination.py

Input files: 2
- PPP_ANN.py
- fatigue_dataset.csv

The fallowing commands will create all the results present in this directory
- python .\PPP_ANN_createOverallresults.py --makeplot 1
- python .\PPP_ANN_combination.py
- python .\PPP_ANN_createOverallresults.py --predictor ON
----------------------------------------------------------------------------------------------------

# Details regarding the files and directories created can be referred from "Table 15" in the report.

File name nomenclature for the files created
Example: plot_testingresults_MSE_SGD --> filename_x_y  (x,y refers to the combination)

x --> Loss
y --> Optimizer

x    | Loss     
MSE  | Mean Squared Error
RMSE | Root Mean Squared Error

y    | Optimizer
SGD  | Stochastic Gradient Descent
SGDM | Stochastic Gradient Descent with momentum
RMSP | Root Mean Squared Propagation
Adam | Adaptive Momentum

Analysis:
--------
Result analysis is performed on the created .xlsx file present in the directory Results_TargetVSpred and saved
as resultcomparisonANN_Analysed.xlsx. The formulas present in this file shall be maintained. The only change
required is adjusting the row entry in the formulas with respect to the number of samples(n) in the test set.

For example:
The results of column J can be created by copying and pasting the formula present in it for each row with respect to the size of the test set. 
For the formulas present in row 100 to 107, column D, the row entry is J3:Jn. Here, n is the number of samples in the test set + 2.