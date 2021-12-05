import context

from sklearn.metrics import precision_recall_fscore_support


if __name__ == '__main__':
    prediction = [0, 0, 1, 1, 0, 1, 0, 1]
    label =      [0, 0, 1, 0, 0, 1, 0, 1]
    p, r, f, s = precision_recall_fscore_support(label, prediction, average='binary')
    print(p)
    print(r)
    print(f)
    print(s)

