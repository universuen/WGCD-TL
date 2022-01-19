import context

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt

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


def cal_cd(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))


excel_path = src.config.path.data / 'kfcv_n_vs_1_result.xlsx'

df = pd.read_excel(excel_path, index_col=0, sheet_name=None)

ranks = {k: {kk: [] for kk in methods} for k in metrics}

for i in metrics:
    scores = {k: [] for k in methods}
    for j in datasets:
        sheet = df[j]
        rank = sheet.T[i].rank(ascending=False)
        for k in methods:
            scores[k].append(sheet[k][i])
            ranks[i][k].append(rank[k])
    print(friedmanchisquare(*scores.values()))
    for j in methods:
        ranks[i][j] = sum(ranks[i][j]) / len(ranks[i][j])

cd = cal_cd(len(datasets), 6, 2.85)
for i in metrics:
    x = list(ranks[i].values())
    y = ranks[i].keys()
    min_ = [i for i in x - cd/2]
    max_ = [i for i in x + cd/2]
    plt.title(f'{i} Friedman Test Result')
    plt.scatter(x, y)
    plt.hlines(y, min_, max_)
    plt.show()

