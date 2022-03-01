import context

if __name__ == '__main__':
    import torch
    import random

    print(
        random.choices(
            population=[1, 2, 3],
            weights=torch.tensor([1, 1, 200]),
            k=20,
        )
    )
