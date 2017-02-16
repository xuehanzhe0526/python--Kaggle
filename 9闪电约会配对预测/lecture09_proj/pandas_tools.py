# -*- coding: utf-8 -*-

'''
Created on 20 Dec, 2016

@author: Robin
'''
import pandas as pd

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
    # 全部转换为小写
    df_data['field'] = df_data['field'].str.lower()
    
    if df_data[['field', 'field_cd']].isnull().values.any():
        # 根据“行业”补全“行业代码”的缺失数据
        
        # 获取“行业”及“行业代码”的唯一值数据
        df_field_and_fcd = df_data[['field','field_cd']].drop_duplicates()
        
        # 删除存在'nan'的行
        df_field_and_fcd = df_field_and_fcd.dropna()
        
        # 将“行业”及“行业代码”生成字典数据
        fcd_dict = dict(zip(df_field_and_fcd['field'], df_field_and_fcd['field_cd']))
        
        # 找出缺失“行业代码”的行索引
        fcd_null_index_lst = df_data[df_data['field_cd'].isnull() & ~df_data['field'].isnull()].index.tolist()
        # 根据字典补全数据
        for fcd_null_index in fcd_null_index_lst:
            field_key = df_data.ix[fcd_null_index, 'field']
            df_data.ix[fcd_null_index, 'field_cd'] = fcd_dict[field_key]
        
        print '补全了%d条“行业代码”缺失数据' % len(fcd_null_index_lst)
        
    return df_data


def get_pair_data(df_data, save_filepath='./processed_dataset.csv'):
    """
            获取重构“成对”数据，以便放入预测模型
    """
    # 使用的特征列表
    used_feat_lst = ['iid', 'gender', 'pid', 'match', 'samerace', 
                     'age_o', 'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int','pf_o_fun', 'pf_o_amb', 'pf_o_sha',
                     'age', 'field_cd', 'race', 'imprace', 'imprelig', 'goal', 'date', 'go_out', 
                     'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
                     'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'exphappy',
                     'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 
                     'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 
                     'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
    
    used_df_data = df_data[used_feat_lst]
    cleaned_df_data = used_df_data.dropna()
    cleaned_df_data = cleaned_df_data.reset_index()
    
    # 根据性别构建男性、女性数据
    m_df_data = cleaned_df_data[cleaned_df_data['gender'] == 1]
    print '男性数据：', m_df_data.shape
    
    f_df_data = cleaned_df_data[cleaned_df_data['gender'] == 0]
    print '女性数据：', f_df_data.shape
    
    # 男性特征列表
    m_feat_lst = ['iid', 'pid', 'match', 'samerace', 'age', 'field_cd', 'race', 
                  'imprace', 'imprelig', 'goal', 'date', 'go_out', 
                  'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
                  'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'exphappy',
                  'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 
                  'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 
                  'attr3_1', 'sinc3_1','fun3_1', 'intel3_1', 'amb3_1']
    new_m_data = m_df_data[m_feat_lst]
    
    # 女性特征列表
    f_feat_lst = [m_feat_lst[0]] + m_feat_lst[4:]
    # 获取唯一的女性数据
    new_f_data = f_df_data[f_feat_lst].drop_duplicates()
    # 重命名女性特征列名，添加前缀'f'
    new_f_data.columns = [ ('f_' + i) for i in f_feat_lst]
    
    print '构建“成对”的数据'
#    # 方法1：循环查找
#     pair_df_data = pd.DataFrame()
#     for _, m_row_data in new_m_data.iterrows():
#         # 对应女性id
#         f_id = m_row_data['pid']
#         # 找到相应id的数据
#         matched_f_data = new_f_data[new_f_data['f_iid'] == f_id]
#         if matched_f_data.shape[0] > 0:
#             # 找到相应的数据
#             pair_row_data = m_row_data.append(matched_f_data.iloc[0])
#             pair_df_data = pair_df_data.append(pair_row_data, ignore_index=True)   
    
    # 方法2： 数据表连接 
    pair_df_data = new_m_data.merge(new_f_data, how='left', left_on='pid', right_on='f_iid')
    # 去除nan数据
    pair_df_data = pair_df_data.dropna()
                
    # 去除id列表
    drop_feat_lst = ['iid', 'pid', 'f_iid']
    pair_df_data = pair_df_data.drop(drop_feat_lst, axis=1)
    
    print '数据重构完成：'
    # 预览数据
    inspect_dataset(pair_df_data)
    
    # 保存数据
    if save_filepath != '':
        pair_df_data.to_csv(save_filepath, index=None)
        
    # 分割“数据”和“标签”
    pair_data = pair_df_data.drop('match', axis=1).values
    labels = pair_df_data['match'].values
    features = pair_df_data.drop('match', axis=1).columns
    
    return pair_data, labels, features