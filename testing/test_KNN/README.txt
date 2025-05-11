========================================================================
# This README.txt file is a part of personal programming project -> test_KNN
# Test_file: test_KNN_Prediction.py
# Test_type: Unit test
Aim: To validate the KNN prediction for a given set of input parameters
------------------------------------------------------------------------

# Expected results:
As it is a unit test, the function is tested using known inputs and pre-calculated outputs.
The details of inputs and outputs can be found in the file: parameters.txt which is present in the same directory.
The pre-calculated outputs will be the expected results and can be found in the column 'expected'.
------------------------------------------------------------------------

# Command used to run the program:
Execute the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details)
# Parameters: 
The parameters for each test case are written in the file: parameters.txt
The program is validated for five different test cases of parameter set and expected output
# Files used: 
parameters.txt, which is present in the same directory
------------------------------------------------------------------------

# limitations:
# When the file parameters.txt is edited for new inputs the fallowing details should be taken into consideration
Remember the input for features is the start, stop value of arange which creates an array of evenly spaced values; dtype int
The length of the range for independent feature and dependent feature should be the same
The length of the expected prediction results will always be 20% of the length of the features
The input for nearest neighbors should be a scalar of dtype int and should be greater than zero
When the entries are edited make sure the alignment between the parameters are maintained
Note that while validating the test the parameters set should always have five test cases with expected results  
------------------------------------------------------------------------

# Obtained result:
pytest framework is used to validate the expected results with the obtained results
Tested using pytest-6.2.5, result: 5 passed -> This confirms that the obtained results is equal to the expected results
========================================================================

