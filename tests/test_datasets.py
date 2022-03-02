import context

from src import utils, datasets

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    utils.prepare_dataset(FILE_NAME)
    # build and test datasets
    full_dataset = datasets.FullDataset()

    positive_dataset = datasets.PositiveDataset()
    assert len(positive_dataset) == sum(positive_dataset.labels), 'all samples in positive dataset should be positive'
    assert len(positive_dataset) == sum(full_dataset.labels), 'all positive samples should be in positive dataset'

    negative_dataset = datasets.NegativeDataset()
    assert sum(negative_dataset.labels) == 0, 'all samples in negative dataset should be negative'
    for sample, label in zip(full_dataset.samples, full_dataset.labels):
        if label == 0:
            assert sample in negative_dataset.samples, 'all negative samples should be in negative dataset'

    weighted_dataset = datasets.WeightedPositiveDataset()
    print(weighted_dataset.get_samples(10000))


