import context

from collections import Counter

import torch
from imblearn.over_sampling import SMOTE

from src import Classifier
from src.dataset import CompleteDataset

if __name__ == '__main__':
    x, y = CompleteDataset(training=True)[:]
    x = x.numpy()
    y = y.numpy()
    print(Counter(y))
    x, y = SMOTE().fit_resample(x, y)
    print(Counter(y))
    smote_dataset = CompleteDataset()
    smote_dataset.features = torch.from_numpy(x)
    smote_dataset.labels = torch.from_numpy(y)
    Classifier('SMOTE_Classifier').train(training_dataset=smote_dataset)