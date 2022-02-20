import pandas as pd
import numpy as np
import math 
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
def load_dataset(filename):
    path= "/Users/wei06159/Desktop/JHU/Machine-Learning/module-3/Programming_Project-2_Sun,Wei-Shan/dataset/" + filename
    column_name = None
    dataset = None
    if "abalone" in filename:
        column_name=['Sex', 'Length', 'Diameter continuous', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        dataset= pd.read_csv(path, names=column_name)
    elif "breast-cancer-wisconsin" in filename:
        column_name=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
        dataset= pd.read_csv(path, names=column_name)
    elif "car" in filename:
        column_name=['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety', 'class']
        dataset= pd.read_csv(path, names=column_name)
    elif "house-votes-84" in filename:
        column_name=['Party','Class Name','handicapped-infants','adoption-of-the-budget-resolution','physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback','education-spendin','superfund-right-to-sue','crime','duty-free-export','export-administration-act-south-africa']
        dataset= pd.read_csv(path, names=column_name)
    elif "forestfires" in filename:
        column_name=['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
        dataset= pd.read_csv(path, names=column_name)
    elif "machine" in filename:
        column_name=['Vendor','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
        dataset= pd.read_csv(path, names=column_name)

    return dataset

# Handling Missing Values - impute missiging data with feature mean
def handling_missing_values(filename,dataset):
    if filename != "house-votes-84.csv":
        
        for (columnName, columnData ) in dataset.iteritems():
            if '?' in str(columnData.values):
                dataset[columnName] = dataset[columnName].replace("?",np.nan)
                dataset[columnName] = dataset[columnName].astype('Int64')
                mean = dataset[columnName].mean().astype(np.int64)
                dataset[columnName] = dataset[columnName].fillna(mean)

    return dataset

# Handling Categorical Data
# Ordinal Data; the define_data is an array with defined order
def ordinal_data(dataset, define_data):
    for i in range(len(define_data)):
        dataset = dataset.replace(define_data[i], i)

    return dataset

# Nominal Data: one-hot encoding: convert data to integer -> transform to new binary
def nominal_data(dataset, target_column):
    # map string to integer
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(dataset[target_column].unique())))])
    binary = []
    for value in dataset[target_column].unique():
        num = str(len(dataset[target_column].unique()))
        temp = format(d[value], "0" + num+"b")
        binary.append(temp)
    target = dataset[target_column].unique()
    for i in range(len(target)):
        column = dataset[target_column].replace(target[i], binary[i])
        dataset = dataset.drop(target_column, axis=1)
        dataset = dataset.join(column)

    return dataset

# Discretization
def discretization(dataset, bin_num, option):

    disc=None
    bins=[]
    # equal-width discretization
    if option == "equal-width discretization":
        for (columnName, _ ) in dataset.iteritems():
            max = dataset[columnName].max()
            min = dataset[columnName].min()
            
            width = int((max-min)/bin_num)

            for i in range(bin_num):
                width = min+ width * (i+1)
                bins.append(width)
            bins.sort()
            dataset[columnName] = np.digitize(dataset[columnName], bins)


        return dataset

    # equal-frequency discretization
    elif option == "equal-frequency discretization":
        length = len(dataset.index)
        num = int(length/bin_num)
        bins=[]

        for (columnName, _ ) in dataset.iteritems():
            if pd.is_numeric_dtype(dataset[columnName]):
                sort_data= dataset[columnName].sort_values()
                for i in range(bin_num):
                    index = num*(i+1)-1
                    bins.append(sort_data.loc[index, columnName])
                dataset[columnName] = np.digitize(dataset[columnName], bins)
                bins=[]
        return dataset

# Standardization
def standardization(training_dataset, test_dataset):

    data_z=(test_dataset-training_dataset.mean())/(training_dataset.std())

    return data_z

# Cross-Validation to handle categorical data with stratify
def cross_validation_w_stratify(dataset, k, stratify_column):
    
    length = len(dataset.index)
    # validation set: 20% from dataset
    val_index_end = int(length * 0.2) # size of validation set
    
    # 80% data for partition
    train_test_dataset=dataset.iloc[val_index_end:, :]
    stratify = train_test_dataset[stratify_column]
    # array containing all unique value from stratifed column
    unique_label_value = stratify.unique()
    
    #split into sub dataset based on unique class label value
    unique_dfs=[]
    for value in unique_label_value:
        temp = dataset[dataset[stratify_column]==value]
        temp=temp.reset_index()
        unique_dfs.append(temp)

    # create validation data set 
    dummy=0
    for value in dataset[stratify_column].value_counts():
        dummy = dummy + value
    num_unique_val = []
    for value in dataset[stratify_column].value_counts():
        num_unique_val.append(int(value * (val_index_end/dummy)))
    
    validation_dataset = pd.DataFrame()
    for i in range(len(num_unique_val)):
        validation_dataset = validation_dataset.append(unique_dfs[i].iloc[0:(num_unique_val[i]+1), :])
        unique_dfs[i]= unique_dfs[i].drop(range(0,(num_unique_val[i]+1)))

    validation_dataset = validation_dataset.reset_index()

    # stratify k fold
    sub_folder = []
    # based on each unique values, split evenly to k group
    for value in unique_dfs:
        temp = np.array_split(value,k)
        sub_folder.append(temp)

    # now, sub_folder stores array values; each array is sub dataset filtered by unique values
    # try to grab each sub dataset from array and append to a full fold
    dfs=[]
    for i in range(k):
        temp = pd.DataFrame()
        for subdata in sub_folder:
            temp=temp.append(subdata[i], ignore_index=True)
        dfs.append(temp)

    return validation_dataset, dfs

# Cross-Validation to handle for regression
def cross_validation(dataset, k):

    length = len(dataset.index)
    val_index_end = int(length * 0.2)
    # validation set: 20% from dataset
    validation_dataset = dataset.iloc[:val_index_end, :]
    # 80% data for partition
    train_test_dataset=dataset.iloc[val_index_end:, :]
    size_fold = int((length * 0.8)/k)
    validation_dataset = validation_dataset.reset_index()
    dfs=[]
    temp_index = 0
    temp=pd.DataFrame()
    for i in range(k):
        if i == (k-1):
            temp = train_test_dataset.iloc[temp_index:, :]
            dfs.append(temp)
            break
        temp = train_test_dataset.iloc[temp_index:(temp_index + size_fold), :]
        temp = temp.reset_index()
        dfs.append(temp)
        temp_index = temp_index + size_fold

    return validation_dataset, dfs


# Evaluation Metrics
# Classification report
# precision, recall, and the F1 score for classification tasks 
def evaluation_metrics_classification(set_true_positive_values, set_true_negative_values, set_predicted_positive_values, set_predicted_negative_values):

    # calcualte precision
    precision = set_true_positive_values/set_predicted_positive_values

    # calculate recall
    recall= set_true_positive_values/(set_true_positive_values + set_predicted_negative_values-set_true_negative_values)

    # F1
    f1= 2 * (precision*recall)/(precision + recall)

    return precision, recall, f1

# Calculate mean absolute error, r2 coefficient
def evaluation_metrics_regression(set_true_values, set_prediction_values):
    # mean absolute error
    length = len(set_true_values)
    mae=0
    for i in range(length):
        mae = mae + abs(set_prediction_values[i]-set_true_values[i])/(length)

    # r2 coefficient
    ssr=0
    tss=0
    mean = np.mean(set_true_values)
    for i in range(length):
        ssr = ssr + math.pow((set_true_values[i]-set_prediction_values[i]), 2)
        tss = tss + math.pow((set_true_values[i]-mean), 2)

    r_square = 1- ssr/tss

    return mae, r_square


# Test:
#handling missing values
# filename="breast-cancer-wisconsin.csv"
# dataset = load_dataset(filename)
# final_dataset = handling_missing_values(filename,dataset)


# # # K-Fold Cross Validation
# i=1
# validation, result = cross_validation_w_stratify(final_dataset, 5, "Class")
# for dataset_new in result:
#     name="fold"+str(i)+".csv"
#     # print(name)
#     # print(dataset_new["Class"].value_counts())
#     i=i+1

# # print(validation)
# # print(validation["Class"].value_counts())
# test = validation["Class"].values
# for i in range(len(test)):
#     print(test[i])
