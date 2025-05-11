========================================================================
# This README.txt file is a part of personal programming project -> test_PCA
# Test_file: test_PCA.py
# Test_type: Testing with sample dataset
Aim: To check the correctness of implementation the model is tested with known dataset and verified whether it is able to recreate the expected results 
------------------------------------------------------------------------

# Expected results:
Refer directory: 'expected_results', which is present in the same directory as README
Expected results are Biplot -> 'expected_plot_PCA.png', Scree plot -> 'expected_plot_Scree.png' and PCA results -> 'expected_results.txt'
PCA: We expect the obtained results and plots present in the directory 'test_results' to be same as the results and plots present in 'expected_results'
Eigenvalues: Checking whether the sum of calculated eigen values are equal to the total variance of the input data
Eigenvectors: Checking whether the length of the computed eigen vector is 1  
------------------------------------------------------------------------

# Command used to run the program:
Run the test from the current directory as pytest -l (prints less details) / pytest -v (prints more details) [Only for default inputs of click]
To test for different inputs(Ex:datasets) it can also be executed as -> python .\test_PCA.py --option
# Files used:
Dataset: sample-data_pca.csv (other dataset can also be used) 
library: PPP_PCA
# Options:
data: Input dataset in .csv format with dependent feature as the last column 
plt_size: Plot size for Biplot and Scree plot; dtype -> float
For more detail regarding the options execute the code as -> python .\test_PCA.py --help 
------------------------------------------------------------------------

# Obtained result:
# if executed as pytest -l / pytest -v :
pytest framework is used to validate the calculated eigen values and eigen vectors
Tested using pytest-6.2.5, result: 1 passed -> This confirms that the obtained results is equal to the expected results
# if executed as python .\test_PCA.py --option / python .\test_PCA.py :
Program prints output_1 -> AssertionPassed: Sum of calculated eigenvalues X == total variance of the data X (Where X denotes the output value; dtype float)
Program prints output_2 -> The length of the computed eigen vector is = 1
Biplot, Scree plot and test results are created and stored in the directory 'test_results'. This then verified with the expected results. 
------------------------------------------------------------------------

# Note:
All results and plots present in the directory: 'expected_results' are from reference[1]
# Reference:
[1] Hartmann, K., Krois, J., Waske, B. (2018): E-Learning Project SOGA: Statistics and Geospatial Data Analysis. 
    Department of Earth Sciences, Freie Universitaet Berlin.
    URL: https://www.geo.fu-berlin.de/en/v/soga/Geodata-analysis/Principal-Component-Analysis/index.html(visited on 22/02/2022)
========================================================================

