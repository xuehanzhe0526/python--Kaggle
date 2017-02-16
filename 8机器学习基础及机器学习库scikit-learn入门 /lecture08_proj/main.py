# -*- coding: utf-8 -*-

"""
Created on Dec 10, 2016

@author: Bin Liang
"""
import pandas as pd
from pandas_tools import inspect_dataset
from pandas_tools import process_missing_data
from pandas_tools import visualize_two_features, visualize_single_feature, \
    visualize_multiple_features
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


def run_main():
    """
        主函数
    """
    # Step.0 加载数据
    filepath = './dataset/voice.csv'
    voice_data = pd.read_csv(filepath)

    # Step.1 查看数据
    inspect_dataset(voice_data)
    # 查看各label的数据量
    print voice_data['label'].value_counts()

    # Step.2 处理缺失数据
    voice_data = process_missing_data(voice_data)

    # Step.3 特征分布可视化
    fea_name1 = 'meanfun'
    fea_name2 = 'centroid'
    visualize_two_features(voice_data, fea_name1, fea_name2)

    visualize_single_feature(voice_data, fea_name1)

    fea_names = ['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']
    visualize_multiple_features(voice_data, fea_names)

    # Step.4 准备数据
    X = voice_data.iloc[:, :-1].values
    voice_data['label'].replace('male', 0, inplace=True)
    voice_data['label'].replace('female', 1, inplace=True)
    y = voice_data['label'].values

    # 特征归一化
    X = preprocessing.scale(X)

    # 分割训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=5)

    # 选择模型，交叉验证
    k_range = range(1, 31)
    cv_scores = []
    print '交叉验证：'
    for k in k_range:
        knn = KNeighborsClassifier(k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        score_mean = scores.mean()
        cv_scores.append(score_mean)
        print '%i: %.4f' % (k, score_mean)

    best_k = np.argmax(cv_scores) + 1
    print '最优K:', best_k

    plt.plot(k_range, cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

    # 训练模型
    knn_model = KNeighborsClassifier(best_k)
    knn_model.fit(X_train, y_train)
    print '测试模型，准确率：', knn_model.score(X_test, y_test)


if __name__ == '__main__':
    run_main()
