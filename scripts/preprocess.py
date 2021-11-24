import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import path


FILE_NAME = 'heart_disease_health_indicators_BRFSS2015.csv'


if __name__ == '__main__':
    # concatenate the file path
    file_path = path.data / FILE_NAME

    # read raw data
    df = pd.read_csv(file_path)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)

    # partition labels and features
    labels = np_array[:, 0].copy().astype('int')
    features = np_array[:, 1:].copy().astype('int')

    # normalize features
    features = MinMaxScaler().fit_transform(features)

    # save to files
    np.save(path.data / 'labels.npy', labels)
    np.save(path.data / 'features.npy', features)
