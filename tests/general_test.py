import torch


if __name__ == '__main__':
    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([4, 5, 6])
    print(sum(abs(a - b)))
