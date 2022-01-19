import context

from math import nan

import pandas as pd
import numpy as np

import src

methods = [
    'Original',
    'RandomOverSampler',
    'SMOTE',
    'ADASYN',
    'BorderlineSMOTE',
    'SNGANHL_EGD',
]

metrics = [
    'F1',
    'G-Mean',
    'AUC',
]

datasets = [
    'ecoli-0-1-4-7_vs_2-3-5-6',
    'ecoli1',
    'ecoli3',
    'glass-0-1-4-6_vs_2',
    'glass-0-1-5_vs_2',
    'glass-0-1-6_vs_2',
    'glass2',
    'glass4',
    'glass5',
    'haberman',
    'page-blocks-1-3_vs_4',
    'pima',
    'poker-8-9_vs_5',
    'winequality-red-8_vs_6',
    'winequality-white-3-9_vs_5',
    'winequality-white-9_vs_4',
    'wisconsin',
    'yeast-0-2-5-6_vs_3-7-8-9',
    'yeast1',
    'yeast-1-2-8-9_vs_7',
    'yeast-1-4-5-8_vs_7',
    'yeast-2_vs_4',
    'yeast4',
    'yeast5',
    'yeast6',
]

excel_path = src.config.path.data / 'kfcv_n_vs_1_result.xlsx'
df = pd.read_excel(excel_path, index_col=0, sheet_name=None)

result = {k: {kk: [] for kk in metrics} for k in methods}
for i in metrics:
    for j in datasets:
        sheet = df[j]
        for k in methods:
            if not np.isnan(sheet[k][i]):
                result[k][i].append(sheet[k][i])
    for j in methods:
        result[j][i] = sum(result[j][i]) / len(result[j][i])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(pd.DataFrame(result))
