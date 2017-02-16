# -*- coding: utf-8 -*-

'''
Created on 20 Dec, 2016

@author: Robin
'''

from sklearn import linear_model, svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.ranking import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score
import numpy as np
from sklearn.model_selection._search import GridSearchCV
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from sklearn.feature_selection.univariate_selection import SelectPercentile
import pandas as pd


def select_features(pair_data, labels, features):
    """
            进行特征选择
    """
    print '特征选择...'
    
    # 1. 过滤掉“低方差”的特征列
    vt_sel = VarianceThreshold(threshold=(0.85 * (1 - 0.85)))
    vt_sel.fit(pair_data)
    
    # 本次实验中没有需要过滤的特征，在这里只是举例
    sel_features1 = features[vt_sel.get_support()]
    sel_pair_data1 = pair_data[:, vt_sel.get_support()]
    print '“低方差”过滤掉%d个特征' % (features.shape[0] - sel_features1.shape[0])
    
    # 2. 根据“单变量统计分析”选择特征\
    # 保留重要的前90%的特征
    sp_sel = SelectPercentile(percentile=95)
    sp_sel.fit(sel_pair_data1, labels)
    
    sel_features2 = sel_features1[sp_sel.get_support()]
    sel_pair_data2 = sel_pair_data1[:, sp_sel.get_support()]
    print '“单变量统计分析”过滤掉%d个特征' % (sel_features1.shape[0] - sel_features2.shape[0])
    
    # 根据特征的score绘制柱状图
    feat_ser = pd.Series(data=sp_sel.scores_, index=features)
    sorted_feat_ser = feat_ser.sort_values(ascending=False)
    plt.figure(figsize=(18, 12))
    sorted_feat_ser.plot(kind='bar')
    plt.savefig('./feat_importance.png')
    plt.show()
    
    return sel_pair_data2, sel_features2
    


def train_model(X_train, y_train, model_name='logistic_regression', is_cv=False):
    """
            训练分类模型，默认为“逻辑回归”模型，默认不执行交叉验证
    """
    if model_name == 'logistic_regression':
        # 逻辑回归   
        lr_model = linear_model.LogisticRegression()
        if is_cv:
            print '交叉验证...'
            params = {'C': [1e-4, 1e-3, 1e-2, 0.1, 1]}
            gs_model = GridSearchCV(lr_model, param_grid=params, cv=5, 
                                    scoring='roc_auc', verbose=3)
            gs_model.fit(X_train, y_train)
            print '最优参数:', gs_model.best_params_
            best_model = gs_model.best_estimator_
        else:
            print '使用模型的默认参数...'
            lr_model.fit(X_train, y_train)
            best_model = lr_model
        
    elif model_name == 'svm':
        # 支持向量机
        svm_model = svm.SVC(probability=True)
        if is_cv:
            print '交叉验证...'
#             params = {'kernel': ('linear', 'rbf'),
#                       'C': [0.01, 0.1, 1, 10, 100]}
            params = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],}
            gs_model = GridSearchCV(svm_model, param_grid=params, cv=5, 
                                    scoring='roc_auc', verbose=3)
            gs_model.fit(X_train, y_train)
            print '最优参数:', gs_model.best_params_
            best_model = gs_model.best_estimator_
        else:
            print '使用模型的默认参数...'
            svm_model.fit(X_train, y_train)
            best_model = svm_model
    
    elif model_name == 'random_forest':
        # 随机森林
        rf_model = RandomForestClassifier()
        if is_cv:
            print '交叉验证...'
            params = {'n_estimators': [20, 40, 60, 80, 100]}
            gs_model = GridSearchCV(rf_model, param_grid=params, cv=5, 
                                    scoring='roc_auc', verbose=3)
            gs_model.fit(X_train, y_train)
            print '最优参数:', gs_model.best_params_
            best_model = gs_model.best_estimator_
        else:
            print '使用模型的默认参数...'
            rf_model.fit(X_train, y_train)
            best_model = rf_model
        
    else:
        # 可以自己添加更多模型用于学习
        print '暂不支持该模型...'
    
    return best_model


def balance_samples(pair_data, labels):
    """
            平衡数据集 
    """
    labels = labels.reshape((labels.size, 1))
    
    all_data = np.concatenate((pair_data, labels), axis=1)
    pos_data = all_data[all_data[:,-1] == 1]
    neg_data = all_data[all_data[:,-1] == 0]
    
    n_pos_samples = pos_data.shape[0]
    
    # 已知负样本过多
    n_selected_neg_samples = int(n_pos_samples * 2)
    sampled_neg_data = neg_data[np.random.choice(neg_data.shape[0], n_selected_neg_samples)]
    
    sampled_all_data = np.concatenate((sampled_neg_data, pos_data))
    
    selected_pair_data = sampled_all_data[:,:-1]
    selected_labels = sampled_all_data[:, -1]
    
    return selected_pair_data, selected_labels
    

def print_test_results(true_labels, pred_labels, pred_probs):
    """
            输出预测结果，包括准确率和AUC值
    """
    print '预测准确率：%.2f' % accuracy_score(true_labels, pred_labels)
    print '预测AUC值：%.4f' % roc_auc_score(true_labels, pred_probs[:, 1])
    print 


def plot_roc(true_labels, pred_probs, fig_title='', savepath=''):
    """
            根据预测值和真实值绘制ROC曲线
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, pred_probs[:, 1], pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.figure()
    plt.title(fig_title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if savepath != '':
        plt.savefig(savepath)
    plt.show()
    