def get_nn(hp):
    # name
    if 'name' in hp['net']:
        name = hp['net']['name']
    else:
        name = hp['net']
    # param
    param = {}
    if 'param' in hp['net']:
        param = hp['net']['param']
    # get neural network object
    if name == 'MLP':
        from .mlp import MLP
        nn_obj = MLP
    elif name == 'CNN':
        from .cnn import CNN
        nn_obj = CNN
    else:
        raise ValueError(f"Unknown neural network name: {name}")
    return nn_obj(**param)