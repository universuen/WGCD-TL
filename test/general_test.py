import context

import random

import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np

from src.config import path
from src import utils

MODELS = (
    'SNGAN_G',
    'SNGAN_EGD',
    'WGANGP_G',
    'WGANGP_EGD',
)
METRICS = (
    'Precision',
    'Recall',
    'F1',
    'Accuracy',
    'AUC',
)


def highlight_higher_cell(s: pd.Series):
    result = []
    for i_1, i_2 in zip(s[0::2], s[1::2]):
        if i_1 > i_2:
            result.append('background-color: yellow')
            result.append('')
        elif i_1 < i_2:
            result.append('')
            result.append('background-color: yellow')
        else:
            result.append('')
            result.append('')
    return result


if __name__ == '__main__':
    utils.set_random_state()
    data = {
        k: {
            kk: random.random() for kk in METRICS
        } for k in MODELS
    }
    df = pd.DataFrame(data)
    df.style.apply(highlight_higher_cell, axis=1)
    df.to_excel('test.xlsx', engine='openpyxl')
