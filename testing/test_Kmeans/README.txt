========================================================================
# This README.txt file is a part of personal programming project -> test_KMeans
# Test_file: test_KMeans_Clustering.py
# Test_type: Testing with sample datasets
Aim: To check the correctness of implementation the model is tested with known datasets(4) and 
verified whether it is able to recreate the expected clusters 
------------------------------------------------------------------------

# Expected results:
Refer directory: 'expected_results', which is present in the same directory as README
Expected result for K-mean clustering -> 'expected_plot_KmeanClustering.png'
K-means: We expect the obtained cluster plots present in the directory 'test_results' to be same as the cluster plots present in 'expected_results'
------------------------------------------------------------------------

# Command used to run the program:
Run the tests by executing
Dataset 1 -> python .\test_KMeans_Clustering.py
Dataset 2 -> python .\test_KMeans_Clustering.py --data sample-data2_kmeans.csv --k 2
Dataset 3 -> python .\test_KMeans_Clustering.py --data sample-data3_kmeans.csv --k 3
Dataset 4 -> python .\test_KMeans_Clustering.py --data sample-data4_kmeans.csv --k 3
# Files used:
Dataset: sample-data1_kmeans.csv/ sample-data2_kmeans.csv/ sample-data3_kmeans.csv/ sample-data4_kmeans.csv 
library: PPP_KMeans
# Options:
data: Input dataset in .csv format with dependent feature as the last column 
k: Select no of clusters from: plot_ElbowMethod.png, present in the directory: 'test_results'
kfind: End limit for no of clusters in elbow plot 
For more detail regarding the options execute the code as -> python .\test_KMeans_Clustering.py --help 
------------------------------------------------------------------------

# Obtained result:
All obtained results for each dataset are stored in the directory: 'test_results' 
Created K-means clustering plots are varified with the expected cluster plots for each dataset
Creates elbow plot, justifying the selection of number of clusters(k) for each dataset.
Results and summary of K-means clustering is also created 
------------------------------------------------------------------------

# Note:
The clustering plots present in the directory: 'expected_results' are from the below references(arranged in order)
# Reference:
[1] Tao Yao. Introduction to K-means Clustering and its Application. 2021. url: https://medium.com/web-mining-is688-spring-2021/introduction-to-
k-means-clustering-and-its-application-in-customer-shopping-dataset-656dcb0a5d09(visited on 02/26/2022).
[2] Usman Malik. Hierarchical Clustering with Python and Scikit-Learn. 2022. url: https://stackabuse.com/hierarchical-clustering-with-python-and
-scikit-learn/(visited on 02/26/2022).
[3] MathWorks. kmeans. 2022. url: https://ch.mathworks.com/help/stats/kmeans.html(visited 03/10/2022).
[4] Data to Fish. Example of K-Means Clustering in Python. 2020. url: https://datatofish.com/k-means-clustering-python/(visited 03/10/2022).  

========================================================================