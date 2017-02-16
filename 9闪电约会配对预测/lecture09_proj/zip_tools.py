# -*- coding: utf-8 -*-

'''
Created on 20 Dec, 2016

@author: Robin
'''
import zipfile


def unzip(zip_filepath, dest_path):
    """
            解压zip文件
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(path = dest_path)


def get_dataset_filename(zip_filepath):
    """
            获取数据库文件名
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        return zf.namelist()[0]