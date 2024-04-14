def get_dataset(path, hp):
    name = hp['dataset']['name']
    param = hp['dataset']['param']
    if name == 'lab1':
        from .lab1 import Lab1
        dataset_obj = Lab1
    elif name == 'CIFAR10':
        from .cifar10 import CIFAR10
        dataset_obj = CIFAR10
    else:
        raise ValueError(f"Invalid dataset: {name}")
    return dataset_obj(path, **param)
