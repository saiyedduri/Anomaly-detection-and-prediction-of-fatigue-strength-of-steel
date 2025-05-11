========================================================================
# This README.txt file is a part of personal programming project -> test_ANN
# Test_file: test_ANN_Functions.py
# Test_type: Unit test
# Aim: To validate activation functions, loss functions and coefficient of determination of a neural network
------------------------------------------------------------------------

# Expected results:
As it is a unit test, the function is tested using known inputs and pre-calculated outputs.
The details of inputs and outputs can be found in the file: parameters.txt which is present in the same directory.
The pre-calculated outputs will be the expected results and can be found in the column 'expected'.
------------------------------------------------------------------------

# Command used to run the program:
Execute the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details)
# Parameters: 
The parameters of each test function are written in the file: parameters.txt
The program is validated for two different test cases of parameter set and expected output
# Files used: 
parameters.txt, which is present in the same directory       
------------------------------------------------------------------------

# limitations:
# When the file parameters.txt is edited for new inputs the fallowing details should be taken into consideration
# In general:
The order in which the parameters set are arranged for each function should be maintained
The number of entires and data types of each parameters set should be maintained
When the entries are edited make sure the alignment between the parameters are maintained
In parameters.txt file it is specifically mentioned to enter 0 when there is no entry
 
# In case of activation functions: The no of entries in the inputset and expected results must be equal
# In case of loss functions: The input set is an array, prediction = inputset.shape and no of entries in 
the expected should be equal to no of rows in the array
# In case of regularization loss: The no of entries in the parameter set should not be modified
# In case of coefficient of determination: The shape of the array for the input set and the prediction set should be the same 

------------------------------------------------------------------------

# Obtained result:
pytest framework is used to validate the expected results with the obtained results
Tested using pytest-6.2.5, result: 12 passed -> This confirms that the obtained results is equal to the expected results
------------------------------------------------------------------------

# Note:
The weights, biases and regularization hyperparameters for each dense layers are fixed
The number of neurons in the hidden layers are also fixed
========================================================================

