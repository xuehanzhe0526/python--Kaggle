# -*- coding: utf-8 -*-

"""
Created on Dec 10, 2016

@author: Bin Liang
"""
import seaborn as sns
import matplotlib.pyplot as plt


def inspect_dataset(df_data):
    """
            查看加载的数据基本信息
    """
    print '数据集基本信息：'
    print df_data.info()
    
    print '数据集有%i行，%i列' %(df_data.shape[0], df_data.shape[1])
    print '数据预览:'
    print df_data.head()


def process_missing_data(df_data):
    """
            处理缺失数据
    """
    if df_data.isnull().values.any():
        # 存在缺失数据
        print '存在缺失数据！'
        df_data = df_data.fillna(0.)    # 填充nan
        # df_data = df_data.dropna()    # 过滤nan
    return df_data.reset_index()


def visualize_two_features(df_data, col_label1, col_label2):
    """
        两个特征分布可视化
    """
    g = sns.FacetGrid(df_data, hue="label", size=8)
    g = g.map(plt.scatter, col_label1, col_label2)
    g.add_legend()
    plt.show()


def visualize_single_feature(df_data, col_label):
    """
        单个特征可视化
    """
    sns.boxplot(x="label", y=col_label, data=df_data)

    g2 = sns.FacetGrid(df_data, hue="label", size=6)
    g2.map(sns.kdeplot, col_label)
    g2.add_legend()

    plt.show()


def visualize_multiple_features(voice_data, fea_names):
    """
        多个特征可视化
    """
    sns.pairplot(voice_data[fea_names], hue='label', size=2)
    plt.show()


