# -*- coding: utf-8 -*-

"""
Created on Dec 10, 2016

@author: Bin Liang
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.io import output_file, show
from bokeh.charts import Bar, TimeSeries
from bokeh.layouts import column
from math import pi


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


def add_year_to_data(air_data):
    """
        将原始数据集的date信息提取出“年份”信息，并放入新的一列
    """
    air_data['Date'] = pd.to_datetime(air_data['Date'])
    air_data['Year'] = air_data['Date'].map(lambda x: x.year)

    return air_data


def plot_crashes_vs_year(air_data, method, save_fig=True):
    """
        每年空难数分析
    """

    if method == 'sns':
        # Seaborn 绘图
        plt.figure(figsize=(15.0, 10.0))
        sns.countplot(x='Year', data=air_data)

        # 解决matplotlib显示中文问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.title(u'空难次数 vs 年份')
        plt.xlabel(u'年份')
        plt.ylabel(u'空难次数')
        plt.xticks(rotation=90)

        if save_fig:
            plt.savefig('./output/crashes_year.png')

        plt.show()

    elif method == 'bokeh':
        # Boken 绘图
        p = Bar(air_data, 'Year', title='空难次数 vs 年份',
                plot_width=1000, legend=False, xlabel='年份', ylabel='空难次数')
        p.xaxis.major_label_orientation = pi / 2
        output_file('./output/crashes_year.html')
        show(p)

    else:
        print '不支持的绘图方式！'


def plot_aboard_vs_fatalities_vs_year(air_data, method, save_fig=True):
    """
        乘客数量vs遇难数vs年份分析
    """

    grouped_year_sum_data = air_data.groupby('Year', as_index=False).sum()

    if method == 'sns':
        # Seaborn 绘图
        plt.figure(figsize=(15.0, 10.0))
        sns.barplot(x='Year', y='Aboard', data=grouped_year_sum_data, color='green')
        sns.barplot(x='Year', y='Fatalities', data=grouped_year_sum_data, color='red')

        # 解决matplotlib显示中文问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.title(u'乘客数量vs遇难数vs年份')
        plt.xlabel(u'年份')
        plt.ylabel(u'乘客数量vs遇难数')
        plt.xticks(rotation=90)

        if save_fig:
            plt.savefig('./output/aboard_fatalities_year.png')

        plt.show()

    elif method == 'bokeh':
        # Boken 绘图
        tsline = TimeSeries(data=grouped_year_sum_data,
                            x='Year', y=['Aboard', 'Fatalities'],
                            color=['Aboard', 'Fatalities'], dash=['Aboard', 'Fatalities'],
                            title='乘客数vs遇难数vs年份', xlabel='年份', ylabel='乘客数vs遇难数',
                            legend=True)
        tspoint = TimeSeries(data=grouped_year_sum_data,
                             x='Year', y=['Aboard', 'Fatalities'],
                             color=['Aboard', 'Fatalities'], dash=['Aboard', 'Fatalities'],
                             builder_type='point',
                             title='乘客数vs遇难数vs年份', xlabel='年份', ylabel='乘客数vs遇难数',
                             legend=True)
        output_file('./output/aboard_fatalities_year.html')
        show(column(tsline, tspoint))
    else:
        print '不支持的绘图方式！'


def plot_top_n(air_data, col_name, save_file_path='', top_n=10, save_fig_path=''):
    """
        col_name的top n分析
    """
    grouped_data = air_data.groupby(by=col_name, as_index=False)['Date'].count()
    grouped_data = grouped_data.rename(columns={'Date': 'Count'})
    top_n_grouped_data = grouped_data.sort_values('Count',
                                                  ascending=False).iloc[:top_n, :]
    # 保存分析文件
    if save_file_path != '':
        top_n_grouped_data.to_csv(save_file_path, index=None)

    # 可视化结果
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Count', y=col_name, data=top_n_grouped_data)

    # 解决matplotlib显示中文问题
    plt.title(u'Count vs %s' % col_name)
    plt.xlabel(col_name)
    plt.ylabel('Count')

    if save_fig_path != '':
        plt.savefig(save_fig_path)

    plt.show()
