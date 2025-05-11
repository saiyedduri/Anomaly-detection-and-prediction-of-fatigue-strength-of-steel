========================================================================
# This README.txt file is a part of personal programming project -> test_MLR
# Test_file: test_MLR_Overfitting.py
# Test_type: Overfitting test
Aim: To check the correctness of implementation the model is tested for overfitting
------------------------------------------------------------------------

# Expected results:
As the model is trained and tested using same dataset, the accuracy result of the model after training and testing should be the same
------------------------------------------------------------------------

# Command used to run the program:
Run the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details) [Only for default inputs of click]
To test for different inputs it can also be executed as -> python .\test_MLR_Overfitting.py --option
# Files used:
Dataset: fatigue_dataset.csv (other dataset can also be used) 
library: PPP_MLR
# Options:
data: Input dataset in .csv format with dependent feature as the last column 
no_of_epoch: Training size; dtype -> int
loss: Loss input; dtype -> str
Other hyperparameter options for optimizers
For more detail regarding the options execute the code as -> python .\test_MLR_Overfitting.py --help 
------------------------------------------------------------------------

# Conditions:
while testing for different combination or new inputs make sure to tune the hyperparameters of the optimizer accordingly
------------------------------------------------------------------------

# Obtained result:
# if executed as pytest -l / pytest -v :
pytest framework is used to validate the accuracy of training result with the accuracy of testing result
Tested using pytest-6.2.5, result: 1 passed -> This confirms that the obtained results is equal to the expected results
# if executed as python .\test_MLR_Overfitting.py --option / python .\test_MLR_Overfitting.py :
And if the accuracy of training result == the accuracy of testing result 
Program prints an output -> AssertionPassed: R^2_training X == R^2_testing X  (Where X denotes the output value; dtype float)
------------------------------------------------------------------------

# Note:
The program can be used to check overfitting for any new parameters set
It can also check for two different combinations of loss with one combinaion at a time
Additional dataset: fatigue_Selecteddataset.csv (Dataset with reduced features) can also be used for testing
Execute : python .\test_MLR_Overfitting.py --data fatigue_Selecteddataset.csv
========================================================================