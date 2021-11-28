import context

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import path, dataset


# FILE_NAME = 'segment0.dat'
# SKIP_ROWS = 24
FILE_NAME = 'page-blocks0.dat'
SKIP_ROWS = 15

if __name__ == '__main__':
    # concatenate the file path
    file_path = path.data / FILE_NAME
    
    # read raw data
    df = pd.read_csv(file_path, sep=', ', engine='python', skiprows=SKIP_ROWS, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    
    # partition labels and features
    labels = np_array[:, -1].copy()
    features = np_array[:, :-1].copy()

    # digitalize labels
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')

    # normalize features
    features = MinMaxScaler().fit_transform(features)


    # partition training and test sets
    training_set_size = int(dataset.training_ratio * len(np_array))
    training_labels, test_labels = np.split(labels, [training_set_size])
    training_features, test_features = np.split(features, [training_set_size])

    # save to files
    np.save(path.data / 'training_labels.npy', training_labels)
    np.save(path.data / 'training_features.npy', training_features)
    np.save(path.data / 'test_labels.npy', test_labels)
    np.save(path.data / 'test_features.npy', test_features)

