# -*- coding: utf-8 -*-

"""
Created on Dec 09, 2016

@author: Bin Liang
"""
import pandas as pd
from pandas_tools import inspect_dataset, add_year_to_data, \
     plot_crashes_vs_year, plot_aboard_vs_fatalities_vs_year, plot_top_n


def run_main():
    """
        主函数
    """
    # Step.0 加载数据
    filepath = './dataset/Airplane_Crashes_and_Fatalities_Since_1908.csv'
    air_data = pd.read_csv(filepath)

    # Step.1 查看数据
    inspect_dataset(air_data)

    # Step.2 数据转换
    air_data = add_year_to_data(air_data)

    # Step.3 数据分析及可视化
    # Step. 3.1 空难数vs年份分析
    plot_crashes_vs_year(air_data, 'sns')
    plot_crashes_vs_year(air_data, 'bokeh')

    # Step. 3.2 乘客数量vs遇难数vs年份分析
    plot_aboard_vs_fatalities_vs_year(air_data, 'sns')
    plot_aboard_vs_fatalities_vs_year(air_data, 'bokeh')

    # Step. 3
    plot_top_n(air_data, 'Type', top_n=20,
               save_file_path='./output/top_50_type.csv',
               save_fig_path='./output/top_50_type.png')
    plot_top_n(air_data, 'Operator', top_n=20,
               save_file_path='./output/top_50_operator.csv',
               save_fig_path='./output/top_50_operator.png')


if __name__ == '__main__':
    run_main()
