import context
from src import VAE, EGAN, EGANClassifier


if __name__ == '__main__':
    VAE().train()
    EGAN().train()
    EGANClassifier().train()
