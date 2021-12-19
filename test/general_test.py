import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cnt = 0
    while cnt < 10:
        cnt += 1
        print(f'\r{cnt}', end='')
        plt.figure('test')
        plt.title(str(cnt))
        plt.plot([random.random() for _ in range(cnt)])
        plt.savefig('test.jpg')
        plt.clf()
        # plt.close('test')

    plt.plot([])
    plt.savefig('test.jpg')