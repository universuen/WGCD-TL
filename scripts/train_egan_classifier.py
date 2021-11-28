import context
from src import VAE, GAN, EGANClassifier


if __name__ == '__main__':
    VAE().train()
    GAN().train()
    EGANClassifier().train()
