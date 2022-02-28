import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    sns.set_style()
    names = ['GAN', 'RVGAN', 'WGAN', 'RVWGAN', 'WGANGP', 'RVWGANGP', 'SNGAN', 'RVSNGAN']
    data = {
        'F1': [0.6172, 0.7109, 0.6317, 0.7221, 0.6387, 0.7099, 0.6418, 0.7365],
        'AUC': [0.7831, 0.8320, 0.7911, 0.8383, 0.7959, 0.8279, 0.7946, 0.8424],
        'G-Mean': [0.7146, 0.7966, 0.7189, 0.8053, 0.7347, 0.7914, 0.7266, 0.8110],
    }
    _, axes = plt.subplots(1, 3)
    for axe, (metric_name, values) in zip(axes, data.items()):
        axe.set(title=metric_name)
        axe.bar(names, values, color=['tab:blue', 'tab:orange'])
        axe.tick_params('x', labelrotation=30)
    plt.show()
