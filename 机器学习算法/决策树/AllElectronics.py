#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by xuehz on 2017/3/3


from sklearn.feature_extraction import DictVectorizer # 将字典 转化为 sklearn 用的数据形式 数据型 矩阵
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO


allElectronicsData = open('AllElectronics.csv','rb')
reader = csv.reader(allElectronicsData)

header = reader.next() #['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']

## 数据预处理

featureList = [] #[{'credit_rating': 'fair', 'age': 'youth', 'student': 'no', 'income': 'high'}, {'credit_rating': 'excellent',
labelList = [] #['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

for row in reader:
    labelList.append(row[-1])
    # 下面这几步的目的是为了让特征值转化成一种字典的形式，就可以调用sk-learn里面的DictVectorizer，直接将特征的类别值转化成0,1值
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[header[i]] = row[i]
    featureList.append(rowDict)

# Vectorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX:"+str(dummyX))

print(vec.get_feature_names())
"""
[[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
 [ 0.  0.  1.  1.  0.  1.  0.  0.  1.  0.]

 ['age=middle_aged', 'age=senior', 'age=youth', 'credit_rating=excellent', 'credit_rating=fair', 'income=high', 'income=low', 'income=medium', 'student=no', 'student=yes']
"""


# label的转化，直接用preprocessing的LabelBinarizer方法
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:"+str(dummyY))
print("labelList:"+str(labelList))
"""
dummyY:[[0]
 [0]
 [1]
 [1]
 [1]

 labelList:['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
"""


#criterion是选择决策树节点的 标准 ，这里是按照“熵”为标准，即ID3算法；默认标准是gini index，即CART算法。

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)

print("clf:"+str(clf))
"""
clf:DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
"""

#生成dot文件
with open("allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf, feature_names= vec.get_feature_names(),out_file= f)

#测试代码，取第1个实例数据，将001->100，即age：youth->middle_aged
oneRowX = dummyX[0,:]
print("oneRowX:"+str(oneRowX))
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:"+str(newRowX))

#预测代码
predictedY = clf.predict(newRowX)
print("predictedY:"+str(predictedY))