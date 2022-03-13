import context

from src import config, utils
from src.datasets import FullDataset
from src.classifier import Classifier
from sklearn.metrics import roc_curve
import seaborn as sns
from matplotlib import pyplot as plt

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normal_Train')
    classifier.fit(
        dataset=FullDataset(),
    )
    test_dataset = FullDataset(test=True)
    x, labels = test_dataset.samples.cpu(), test_dataset.labels.cpu()
    predicted_prob = classifier.predict(x, use_prob=True).cpu()
    fpr, tpr, _ = roc_curve(labels, predicted_prob)

    sns.set_style('white')
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig(config.path_config.test_results / 'ROC.jpg')
    plt.show()
