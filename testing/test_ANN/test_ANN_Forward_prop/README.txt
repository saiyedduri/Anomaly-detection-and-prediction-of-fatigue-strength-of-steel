========================================================================
# This README.txt file is a part of personal programming project -> test_ANN
# Test_file: test_ANN_Forward_prop.py
# Test_type: Unit test
# Aim: To validate one complete forward propagation step of a neural network
------------------------------------------------------------------------

# Expected results:
As it is a unit test, the function is tested using known inputs and pre-calculated outputs.
The details of inputs and outputs can be found in the file: parameters.txt which is present in the same directory.
The pre-calculated outputs will be the expected results and can be found in the column 'expected'.
------------------------------------------------------------------------

# Command used to run the program:
Execute the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details)
# Parameters: 
The input set and expected output are written in the file: parameters.txt
The program is validated for five different test cases of input set and expected output
# Files used: 
parameters.txt, which is present in the same directory        
------------------------------------------------------------------------

# limitations:
# When the file parameters.txt is edited for new inputs the fallowing details should be taken into consideration
The input set should always be a single sample with five neurons with dtype:int and atleast one neuron value must be greater than zero
The expected set should also be of dtype:int and it is a scalar
When the entries are edited make sure the alignment between inputset and the expected are maintained
Note that while validating the test the parameters set should always have five input sets and five expected results  
------------------------------------------------------------------------

# Obtained result:
pytest framework is used to validate the expected results with the obtained results
Tested using pytest-6.2.5, result: 5 passed -> This confirms that the obtained results is equal to the expected results
------------------------------------------------------------------------

# Note:
The weights and biases for each dense layers are fixed
The number of neurons in the hidden layers are also fixed
========================================================================

