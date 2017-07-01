import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import csv
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
scaler = StandardScaler()
scaler.fit(df.drop('SITE', axis = 1))
scaled_features = scaler.transform(df.drop('SITE', axis = 1))
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
accuracy_list = []
n_splits=5
kf = KFold(n_splits)
test=[]
train=[]
def getValues(count):
    if count==1:
        for i in range(100):
            if 0<=i<20:
                test.append(i)
            else:
                train.append(i)

    if count==2:
        for i in range(100):
            if 20<=i<40:
                test.append(i)
            else:
                train.append(i)
    if count==3:
        for i in range(100):
            if 40<=i<60:
                test.append(i)
            else:
                train.append(i)
    if count==4:
        for i in range(100):
            if 60<=i<80:
                test.append(i)
            else:
                train.append(i)
    if count==5:
        for i in range(100):
            if 80<=i<100:
                test.append(i)
            else:
                train.append(i)
    return test,train
print('\n')
print("                       THE VARYING K ANALYSIS")
for x in range(5):
    test=[]
    train=[]
    test,train=getValues(x+1)
    pred_train = df_feat.ix[train]
    tar_train = df['SITE'][train]
    pred_test = df_feat.ix[test]
    tar_test = df['SITE'][test]

    # Finding the best value of K with minimum error rate for each fold!!
    error_rate = []
    kmin=[]
    for i in range(1,20):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(pred_train, tar_train)
        pred_i = knn.predict(pred_test)
        error_rate.append(np.mean(pred_i != tar_test))
        kmin.append((error_rate[i-1],i))
    kmin.sort()
    k=kmin[0][1] # minimum K value found!


    # Using the minimum K value to run the algorithm!
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(pred_train, tar_train)
    pred = knn.predict(pred_test)
    a=int(sklearn.metrics.accuracy_score(tar_test,pred)*100)
    print("===========================================================")
    print('Iteration #'+str(x+1)+'\n')
    print("Value of K: "+str(k))
    print("Accuracy is: "+str(a)+'%')
    accuracy_list.append(sklearn.metrics.accuracy_score(tar_test,pred)*100)
    print(confusion_matrix(tar_test,pred))
    print('\n')
    print('The F-Score Report:')
    print(classification_report(tar_test,pred))
l=[int(x) for x in accuracy_list]
print
with open("KNN_Varying_K_Accuracy.csv", "w") as fp_out:
    writer = csv.writer(fp_out, delimiter=",")
    writer.writerow(l)
print("===========================================================")
print("===========================================================")
print ('\n'+'The Average Accuracy after 5 Fold Cross Validation is: '+ str(int(sum(accuracy_list)/len(accuracy_list)))+'%')#'''
import numpy as np
import matplotlib.pyplot as plt
import csv

### VISUALIZING THE ACCURACY DATA ###
# data to plot
n_groups = 5
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
v=[]
for i in range(len(accuracy_list)):
    v.append(int(accuracy_list[i]))
# print(v)
rects1 = plt.bar(index, v, bar_width,
                 alpha=opacity,
                 color='r',
                 label='KNN_Varying_K')
#
# rects2 = plt.bar(index + bar_width, l, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label='Decision Tree')

plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy by KNN_Varying_K')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()
print("===========================================================")
print("===========================================================")
