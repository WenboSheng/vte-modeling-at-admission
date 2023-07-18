#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some utilities
"""
import os
import zipfile
import argparse
from typing import List, Mapping
import joblib
import dill
import yaml
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils import check_random_state


def gen_seed(seed, n):
    """generate a group of random seeds"""
    rng = check_random_state(seed)
    r_state = rng.get_state()
    assert n <= len(r_state[1]), \
        f'too large seed numbers: {n}, should be less than {len(r_state[1])}'
    if n == 1:
        # return scalar
        return rng.randint(np.iinfo(np.int32).max + 1)
    else:
        # return array
        return r_state[1][-n:] // np.iinfo(np.uint16).max


def load_yml(path):
    with open(path, encoding='utf8') as f:
        config = yaml.safe_load(f)
    return config


def check_file_path(path: str):
    if path.find('/') >= 0:
        # 查找最后一个 /
        file_dir = '/'.join(path.split('/')[0:-1])
        if len(file_dir) > 0:
            os.makedirs(file_dir, exist_ok=True)


def save_dataframe_dict(data: Mapping[str, pd.DataFrame], file: str, **kwargs):
    """
    Save a dict of pd.Dataframe to excel.
    :param data: dict of dataframes with keys as excel sheets
    :param file: output file name
    :param kwargs: dict for args of each `to_excel` methods
    :return: None
    """
    check_file_path(file)
    # check args for each sheet
    common_kw = {key: val for key, val in kwargs.items() if key not in data.keys()}
    sheet_kw = dict()
    for _sheet in data.keys():
        sheet_kw[_sheet] = common_kw.copy()
        if _sheet in kwargs.keys():
            sheet_kw[_sheet].update(kwargs.get(_sheet))
    # write excel
    with pd.ExcelWriter(file) as writer:
        for _sheet, _df in data.items():
            _df.to_excel(writer, sheet_name=_sheet, **sheet_kw[_sheet])


def save_pickle(obj, file: str, **kwargs):
    """python object -> pickle file"""
    check_file_path(file)
    kw = kwargs.copy()
    compress = kw.pop('compress') if 'compress' in kw else ('xz', 3)
    protocol = kw.pop('protocol') if 'protocol' in kw else 3
    joblib.dump(obj, file, compress=compress, protocol=protocol, **kwargs)


def read_table_file(file_path: str, **kwargs):
    """read file as a dataframe"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('csv.zip'):
        if 'zipped_file' in kwargs:
            kw1 = kwargs.copy()
            zipped_file = kw1.pop('zipped_file')
        else:
            kw1 = kwargs
            zipped_file = os.path.basename(file_path)[:-4]
        with zipfile.ZipFile(file_path) as zip_obj:
            with zip_obj.open(zipped_file) as fopen:
                return pd.read_csv(fopen, **kw1)
    elif file_path.endswith('.pkl'):
        return load_pickle(file_path)
    else:
        raise TypeError(f'Unsupported file type {file_path}')


def load_pickle(file):
    """pickle file -> python object"""
    assert os.path.isfile(file) and os.access(file, os.R_OK), f'{file} not exists or not readable'
    try:
        obj = joblib.load(file)
    except ValueError:
        obj = dill.load(open(file, 'rb'))
    return obj


def safe_drop_columns(df: pd.DataFrame, columns: List, **kwargs):
    """save drop columns in ca"""
    cols_drop = [col for col in columns if col in df.columns]
    if len(cols_drop) > 0:
        df1 = df.drop(columns=cols_drop, **kwargs)
    else:
        df1 = df.copy()
    return df1


def get_parser(log_file=False):
    """for consistency with terminal / pycharm / jupyter"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='client', type=str)
    parser.add_argument('--port', default=60472, type=int)
    parser.add_argument('-f', default='', type=str)
    if log_file is True:
        parser.add_argument('--log_file', default='', type=str)
    return parser


def log_df_info(df, df_name):
    """Log data frame info (size and type)"""
    logger.debug(f"{df_name} data size: {df.shape}")
    logger.debug(f"{df_name} data type: \n{df.dtypes}")

# EOF
