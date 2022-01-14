import context

import pandas as pd

import src

headers_1 = [
    'GAN',
    'GANHL_EGD',
    'WGAN',
    'WGANHL_EGD',
    'WGANGP',
    'WGANGPHL_EGD',
    'SNGAN',
    'SNGANHL_EGD',
]

headers_2 = [
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

excel_path_1 = src.config.path.data / 'kfcv_g_vs_hl_egd_result.xlsx'
excel_path_2 = src.config.path.data / 'kfcv_n_vs_1_result.xlsx'

df_1 = pd.read_excel(excel_path_1, index_col=0, sheet_name=None)
df_2 = pd.read_excel(excel_path_2, index_col=0, sheet_name=None)

result_1 = []
result_2 = []

for i in df_1:
    sheet = df_1[i]
    cnt_1 = 0
    for j in metrics:
        row = sheet.T[j]
        cnt_2 = 0
        for k_1, k_2 in zip(headers_1[0::2], headers_1[1::2]):
            if row[k_2] > row[k_1]:
                cnt_2 += 1
        if cnt_2 >= 2:
            cnt_1 += 1
    if cnt_1 >= 2:
        result_1.append(i)

for i in df_2:
    sheet = df_2[i]
    cnt = 0
    for j in metrics:
        row = sheet.T[j]
        if row.idxmax() == headers_2[-1]:
            cnt += 1
    if cnt >= 2:
        result_2.append(i)

cnt = 0
for i in result_1:
    if i in result_2:
        cnt += 1
        print(cnt, i)

