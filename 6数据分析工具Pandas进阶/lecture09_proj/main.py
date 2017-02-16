# -*- coding: utf-8 -*-

'''
Created on 20 Dec, 2016

@author: Robin
'''
import os
from zip_tools import get_dataset_filename, unzip
import pandas as pd
from pandas_tools import inspect_dataset, process_missing_data, get_pair_data
from sklearn.model_selection import train_test_split
from ml_tools import train_model, plot_roc, print_test_results, balance_samples,\
    select_features
import numpy as np

# 声明变量
dataset_path = './dataset'    # 数据集路径
zip_filename = 'Speed Dating Data.csv.zip'     # zip文件名
zip_filepath = os.path.join(dataset_path, zip_filename)    # zip文件路径
dataset_filename = get_dataset_filename(zip_filepath)      # 数据集文件名（在zip中）
dataset_filepath = os.path.join(dataset_path, dataset_filename)  # 数据集文件路径

# 是否处理非平衡数据
is_process_unbalanced_data = True

# 是否交叉验证
is_cv = True

# 是否进行特征选择
is_feat_select = True

# 设置随机种子
random_seed = 7
np.random.seed(random_seed)


def run_main():
    """
            主函数
    """
    
    ## 解压数据集
    print "解压zip...",
    unzip(zip_filepath, dataset_path)
    print "完成."
    
    ## 1. 查看数据集
    df_data = pd.read_csv(dataset_filepath)
    inspect_dataset(df_data)
    
    ## 2. 处理缺失数据
    df_data = process_missing_data(df_data)
    
    ## 3. 数据处理构建特征，并重构数据
    # 获取重构“成对”数据，以便放入预测模型
    pair_data, labels, features = get_pair_data(df_data)
    
    # 进行特征选择
    if is_feat_select:
        pair_data, selected_features = select_features(pair_data, labels, features)
        print '选择的特征：',
        print selected_features
    
    n_pos_samples = labels[labels == 1].shape[0]
    n_neg_samples = labels[labels == 0].shape[0]
    print '正样本数：%d' % n_pos_samples
    print '负样本数：%d' % n_neg_samples
    
    # 处理非平衡数据
    if is_process_unbalanced_data:
        pair_data, labels = balance_samples(pair_data, labels)
    
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(pair_data, labels, 
                                                        test_size=0.1, 
                                                        random_state=random_seed)
    
    
    ## 4.训练模型，测试模型
    print "逻辑回归模型："
    logistic_model = train_model(X_train, y_train, 
                                 model_name='logistic_regression', is_cv=is_cv)
    logistic_model_predictions = logistic_model.predict(X_test)
    logistic_model_prob_predictions = logistic_model.predict_proba(X_test)
    # 输出预测结果
    print_test_results(y_test, logistic_model_predictions, logistic_model_prob_predictions)
    
    print "支持向量机模型："
    svm_model = train_model(X_train, y_train, 
                            model_name='svm', is_cv=is_cv)
    svm_model_predictions = svm_model.predict(X_test)
    svm_model_prob_predictions = svm_model.predict_proba(X_test)
    # 输出预测结果
    print_test_results(y_test, svm_model_predictions, svm_model_prob_predictions)
    
    print "随机森林模型："
    rf_model = train_model(X_train, y_train, 
                           model_name='random_forest', is_cv=is_cv)
    rf_model_predictios = rf_model.predict(X_test)
    rf_model_prob_predictios = rf_model.predict_proba(X_test) 
    # 输出预测结果
    print_test_results(y_test, rf_model_predictios, rf_model_prob_predictios)
    
    ## 5. 绘制ROC曲线
    plot_roc(y_test, logistic_model_prob_predictions, 
             fig_title='Logistic Regression', savepath='./lr_roc.png')
    plot_roc(y_test, svm_model_prob_predictions, 
             fig_title='SVM', savepath='./svm_roc.png')
    plot_roc(y_test, rf_model_prob_predictios, 
             fig_title='Random Forest', savepath='./rf_roc.png')
       
    
    # 删除解压数据，清理空间
    if os.path.exists(dataset_filepath):
        print '分析结束，清理空间。'
        os.remove(dataset_filepath)
        

if __name__ == '__main__':
    run_main()