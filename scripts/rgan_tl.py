import src

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    src.utils.prepare_dataset(FILE_NAME)
    rgan_dataset = src.utils.get_rgan_dataset()
    pass
