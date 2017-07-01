import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv
subprocess.call('python KNN.py',shell=True)
subprocess.call('python Decision_Tree.py',shell=True)
# subprocess.call('python KNN_Varying_K.py',shell=True)

os.remove('1.pdf')
os.remove('2.pdf')
os.remove('3.pdf')
os.remove('4.pdf')
os.remove('5.pdf')



with open('KNN_Accuracy.csv') as f1:
    r=csv.reader(f1,delimiter=',')
    for row in r:
        a=row
        #next(r)
    for x in range(len(a)):
        a=[int(x) for x in a]
#
# with open('KNN_Varying_K_Accuracy.csv') as f1:
#     r=csv.reader(f1,delimiter=',')
#     for row in r:
#         b=row
#         next(r)
#     for x in range(len(b)):
#         b=[int(x) for x in b]

with open('DecisionTree_Accuracy.csv') as f2:
    r=csv.reader(f2,delimiter=',')
    for row in r:
        c=row
        #next(r)
    for x in range(len(c)):
        c=[int(x) for x in c]
# print(b)

# data to plot
n_groups = 5
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, a, bar_width,
                 alpha=opacity,
                 color='b',
                 label='KNN')
# rects2 = plt.bar(index + bar_width, b, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  label='KNN_Varying_K')
rects3 = plt.bar(index + bar_width, c, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Arvore de Decisao')


plt.xlabel('Folds')
plt.ylabel('Precisao')
plt.title('Precisao por Algoritmo')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()

os.remove('KNN_Accuracy.csv')
# os.remove('KNN_Varying_K_Accuracy.csv')
os.remove('DecisionTree_Accuracy.csv')
