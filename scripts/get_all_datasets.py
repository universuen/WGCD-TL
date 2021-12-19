import glob
from os.path import basename

from src import config


if __name__ == '__main__':
    all_datasets = [basename(p) for p in glob.glob(str(config.path.data / '*.dat'))]
    target_path = config.path.data / 'all_datasets.txt'
    with open(target_path, 'w') as f:
        for i in all_datasets:
            f.write(f"'{i}',\n")
    print(f'Saved dataset list in {target_path}')
