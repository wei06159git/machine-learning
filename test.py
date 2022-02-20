import knn as k
import toolset as t

# Test:
# breast cancer
filename="breast-cancer-wisconsin.csv"
dataset = t.load_dataset(filename)
dataset = t.handling_missing_values(filename,dataset)
validation, dfs = t.cross_validation_w_stratify(dataset, 5,"Class")
dfs[4].to_csv("fold2.csv")
validation.to_csv("tune.csv")

class_column_name= "Class"
class_column_list = ['Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
predict = k.k_nn(7, dfs[4], validation, class_column_name, None, class_column_list , "classification")

error = k.classification_error(validation, predict, "Class")
print(predict)
print(validation["Class"])
print ("error rate for k ", str(7), " is ", str(error) )