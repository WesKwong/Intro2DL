def get_dataset(path, hp):
    # name
    if 'name' in hp['dataset']:
        name = hp['dataset']['name']
    else:
        name = hp['dataset']
    # param
    param = {}
    if 'param' in hp['dataset']:
        param = hp['dataset']['param']
    # get dataset object
    if name == 'Lab1':
        from .lab1 import Lab1
        dataset_obj = Lab1
    elif name == 'CIFAR10':
        from .cifar10 import CIFAR10
        dataset_obj = CIFAR10
    else:
        raise ValueError(f"Invalid dataset: {name}")
    return dataset_obj(path, **param)
