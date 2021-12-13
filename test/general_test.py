import numpy as np
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':
    raw_x = np.random.random([100, 1])
    raw_y = np.random.randint(0, 2, 100)
    x, y = SMOTE().fit_resample(raw_x, raw_y)
    pass

