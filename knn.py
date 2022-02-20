# Programming Project #2

from dis import dis
from hmac import new
import pandas as pd
import numpy as np
import math
from toolset import cross_validation_w_stratify, cross_validation, load_dataset, ordinal_data
from operator import itemgetter

# k-nearest neighbor

# find value k

# Calculate Euclidean distance between numeric data
# point 1 is training data
# point 2 is tuning/test data
def euclidean_distance(point1, point2, numeric_list):
    distance=0
    for i in range(len(numeric_list)): # does not include the last column which in each row is an output value
        distance += pow((point1[numeric_list[i]].values - point2[numeric_list[i]].values),2)
    distance = math.sqrt(distance)

    return distance

# Calculate distance between categorical data using value difference metric; point1 is from training dataset
# point 1 is training data
# point 2 is tuning/test data
def value_diff_metric(point1, point2, column, training_data, class_label, class_column):
    distance = 0

    for i in range(len(class_label)):
        i_value = point1[column].values[0]
        j_value = point2[column].values[0]

        ci_a = len(training_data[(training_data[column]==i_value) & (training_data[class_column]==class_label[i])])
        cj_a = len(training_data[(training_data[column]==j_value) & (training_data[class_column]==class_label[i])])
        ci = len(training_data[training_data[column]==i_value])
        cj = len(training_data[training_data[column]==j_value])
        if cj ==0:
            distance = distance + abs(ci_a/ci - 0) 
        else:
            distance = distance + abs(ci_a/ci - cj_a/cj) 
    return distance

# Calculate combined distance (categorical and numerical features)
def combined_distance(numerical_distance, categorical_distance):
    distance=0
    distance = numerical_distance + categorical_distance

    return distance

# do cross validation for tuning to pick best hyper-parameter (find k value) (20% data for tuning and need to be stratified for classification)
# pick k to minimize the loss

# tune k

# k-nn algorithm
def k_nn(k, training_data, test_data, class_column_name, numerical_column_list, categorical_column_list, option):
    numeric_dist=0.0
    cat_dist=0.0
    dist=[] # distance and class label
    predicted_list = []
    for i in range(len(test_data)):
        test_row = test_data.iloc[i:i+1, :]
        print("point " + str(i+1))
        for b in range(len(training_data)):
            training_row = training_data.iloc[b:b+1, :]
            if numerical_column_list is not None:
                numeric_dist= euclidean_distance(test_row, training_row, numerical_column_list)
            if categorical_column_list is not None:
                for j in range(len(categorical_column_list)):
                    class_label = training_data[class_column_name].unique()
                    cat_dist = cat_dist + value_diff_metric(training_row, test_row, categorical_column_list[j], training_data, class_label, class_column_name)
            distance = numeric_dist + cat_dist
            temp=[distance, training_row[class_column_name].values[0]]
            dist.append(temp)
            numeric_dist = 0 #reset
            cat_dist = 0 #reset

        if option == "classification":
            predict = prediction_class_knn(dist,k)
            predicted_list.append(predict)
        elif option == "regression":
            predict = predict_regression_knn(dist,k)
            predicted_list.append(predict)

    return predicted_list

# employ plurality vote to determine class
def prediction_class_knn(dist,k):
    dist = sorted(dist, key=itemgetter(0))
    new_dist = dist[0:k]
    new_class_list =[]
    for value in new_dist:
        new_class_list.append(value[1])

    predicted_value = max(new_class_list, key = new_class_list.count)
    return predicted_value

# apply a Gaussian (radial basis function) kernel to make prediction
def predict_regression_knn(dist, k):
    new_dist=[]
    dist_list=[]
    temp=0
    # for i in range(len(dist)):
    #     dist_list.append(dist[i][0])
    # test_signma = np.square(np.std(dist_list))
    # for i in range(len(dist)):
    #     temp = np.exp(-1*dist[i][0]/test_signma)
    #     new_dist.append([temp, dist[i][1]])
        
    new_dist = sorted(dist, key=itemgetter(0))
    g_dist = new_dist[(len(new_dist)-k):(len(new_dist))]
    sum=0
    for i in range(len(g_dist)):
        sum=sum + g_dist[i][1]
        print(dist[i][1])
    predicted = sum/len(g_dist)
    return predicted

# Calculate classification error (FP+FN)/(P+N)
def classification_error(test_data, predict, clas_column_name):
    error_count = 0
    test_data = test_data[clas_column_name].values
    for i in range(len(test_data)):
        if test_data[i] != predict[i]:
            error_count = error_count + 1
    error_rate = error_count/(len(test_data))
    return error_rate

# Calculate mean squared error for regression 1/n*(sigma x-y)^2
def regression_error(test_data, predict, clas_column_name):
    difference_square = 0
    test_data = test_data[clas_column_name].values
    for i in range(len(test_data)):
        difference_square = difference_square + math.pow((test_data[i]-predict[i]),2)

    mean_square_error = difference_square/(len(test_data))
    return mean_square_error

# edited k-nearest neighbor; tune k and epsilon
# Reduce irrelevant attribute
def edited_knn(k, training_data, test_data, class_column_name, numerical_column_list, categorical_column_list, option):
    k=1
    predict_edited = k_nn(k, training_data, test_data, class_column_name, numerical_column_list, categorical_column_list, option)
    class_column_index = training_data.colimns.get_loc(class_column_name)
    for i in range(len(training_data)):
        if predict_edited[i] != training_data.iloc[i:(i+1), class_column_index:(class_column_index+1)]:
            training_data = training_data.drop(labels=i, axis=0)

    return training_data

#condensed k-nearest neighbor; tune k and epsilon
def condensed_knn(k, training_data, test_data, class_column_name, numerical_column_list, categorical_column_list, option):
    k=1
    predict_edited = k_nn(k, training_data, test_data, class_column_name, numerical_column_list, categorical_column_list, option)
    z_set = pd.DataFrame()
    class_column_index = training_data.colimns.get_loc(class_column_name)
    for i in range(len(training_data)):
        if predict_edited[i] != training_data.iloc[i:(i+1), class_column_index:(class_column_index+1)]:
            z_set = z_set.append(training_data.iloc[i:(i+1), class_column_index:(class_column_index+1)])
    return z_set

# Get Epsilon for regression; the maximum distance that still have same class. Just an error threshold to determine if a prediction is correct
def epsilon(training_data, test_data, class_column_name, numerical_column_list):
    distance = 0
    numeric_dist=0
    dist=[] # distance and class label
    for i in range(len(test_data)):
        test_row = test_data.iloc[i:i+1, :]
        for b in range(len(training_data)):
            training_row = training_data.iloc[b:b+1, :]
            numeric_dist= numeric_dist + euclidean_distance(test_row, training_row, numerical_column_list)
            
            distance = numeric_dist
            temp=[distance, training_row[class_column_name], test_row[class_column_name]]
            dist.append(temp)