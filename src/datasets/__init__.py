from numpy import ndarray

from src.datasets.dataset_like import DatasetLike

from src.datasets.full_dataset import FullDataset
from src.datasets.negative_dataset import NegativeDataset
from src.datasets.positive_dataset import PositiveDataset
from src.datasets.roulette_positive_dataset import RoulettePositiveDataset


training_samples: ndarray = None
training_labels: ndarray = None
test_samples: ndarray = None
test_labels: ndarray = None
