import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
data_base_name = "isolet.csv"
df = pd.read_csv(data_base_name, encoding = "ISO-8859-1")
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
        for i in range(150):
            if 0<=i<30:
                test.append(i)
            else:
                train.append(i)
    if count==2:
        for i in range(150):
            if 30<=i<60:
                test.append(i)
            else:
                train.append(i)
    if count==3:
        for i in range(150):
            if 60<=i<90:
                test.append(i)
            else:
                train.append(i)
    if count==4:
        for i in range(150):
            if 90<=i<120:
                test.append(i)
            else:
                train.append(i)
    if count==5:
        for i in range(150):
            if 120<=i<150:
                test.append(i)
            else:
                train.append(i)
    return test,train
print('\n')
km=[]
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
    plt.figure(figsize=(10,6))
    plt.plot(range(1,20),error_rate, color = 'red',linestyle = 'dashed', marker = 'o', markerfacecolor='red',markersize=10)
    plt.title('Taxa de Erro vs. Valor de K')
    plt.xlabel('Valor de K')
    plt.ylabel('Taxa de Erro')
    plt.show()
    kmin.sort()
    ka=kmin[0][1] # minimum K value found!
    km.append(ka)
    # print(km)
    kmsum=0
    kmsum=sum(km)
    # print(kmsum)
    k=math.ceil(kmsum/5)
    # print(k)
print("                      Analise do KNN")
for x in range(5):
    test=[]
    train=[]
    test,train=getValues(x+1)
    pred_train = df_feat.ix[train]
    tar_train = df['SITE'][train]
    pred_test = df_feat.ix[test]
    tar_test = df['SITE'][test]
    # Using the minimum K value to run the algorithm!
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(pred_train, tar_train)
    pred = knn.predict(pred_test)
    a=int(sklearn.metrics.accuracy_score(tar_test,pred)*100)
    print("===========================================================")
    print('Iteracao #'+str(x+1)+'\n')
    print("Valor de K: "+str(k))
    print("Precisao: "+str(a)+'%')
    accuracy_list.append(sklearn.metrics.accuracy_score(tar_test,pred)*100)
    print(confusion_matrix(tar_test,pred))
    # print('\n')
    print('O F-Score:')
    print(classification_report(tar_test,pred))
l=[int(x) for x in accuracy_list]
print
with open("KNN_Accuracy.csv", "w") as fp_out:
    writer = csv.writer(fp_out, delimiter=",")
    writer.writerow(l)
print("===========================================================")
print("===========================================================")
print ('\n'+'A media de precisao apos 5 Fold Cross Validation e: '+ str(int(sum(accuracy_list)/len(accuracy_list)))+'%')
print("===========================================================")
print("===========================================================")
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
rects1 = plt.bar(index, v, bar_width,
                 alpha=opacity,
                 color='b',
                 label='KNN')
plt.xlabel('Folds')
plt.ylabel('Precisao')
plt.title('Precisao com KNN')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()
