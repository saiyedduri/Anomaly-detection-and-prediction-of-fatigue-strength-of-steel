========================================================================
# This README.txt file is a part of personal programming project -> test_ANN
# Test_file: test_CompleteANN_SGD_optimizer.py
# Test_type: Gradient checking
Aim: To validate one complete training step using SGD optimizer
------------------------------------------------------------------------

# Expected results:
Back propagation: Numerical approximation of the derivative (gradient checking) should satisfy the originally computed derivative
SGD optimizer: Optimization computed using numerical gradient results should satisfy the originally computed optimization
Here, optimization means updating weights and biases 
------------------------------------------------------------------------

# Command used to run the program:
Run the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details)
# Parameters: 
The input set and dependent feature are written in the file: parameters.txt
The program is validated for five different test cases as seen in parameters.txt file
# Files used: 
parameters.txt, which is present in the same directory
------------------------------------------------------------------------

# limitations:
# When the file parameters.txt is edited for new inputs the fallowing details should be taken into consideration
The input set should always be a single sample with five neurons with dtype:int and atleast one neuron value must be greater than zero
The dependent feature should be of dtype:float and it is a scalar
When the entries are edited make sure the alignment between the parameters are maintained
Note that while validating the test the parameters set should always have five input sets and dependent features  
------------------------------------------------------------------------

# Obtained result:
pytest framework is used to validate the expected results with the obtained results
Tested using pytest-6.2.5, result: 5 passed -> This confirms that the obtained results is equal to the expected results
------------------------------------------------------------------------

# Note:
The weights and biases for each dense layers are fixed
The number of neurons in the hidden layers are fixed
The hyperparameters for the SGD optimizer are also fixed
========================================================================

