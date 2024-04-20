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
    # ------------- get neural network object ------------ #
    if name == 'MLP':
        from .mlp import MLP
        nn_obj = MLP
    elif name == 'LeNet5':
        from .cnn import LeNet5
        nn_obj = LeNet5
    elif name == 'BNormLeNet5':
        from .cnn import BNormLeNet5
        nn_obj = BNormLeNet5
    elif name == 'LNormLeNet5':
        from .cnn import LNormLeNet5
        nn_obj = LNormLeNet5
    elif name == 'INormLeNet5':
        from .cnn import INormLeNet5
        nn_obj = INormLeNet5
    elif name == 'PadBNLeNet5':
        from .cnn import PadBNLeNet5
        nn_obj = PadBNLeNet5
    elif name == 'PadBNKernel3LeNet5':
        from .cnn import PadBNKernel3LeNet5
        nn_obj = PadBNKernel3LeNet5
    elif name == 'DropoutPadBNLeNet5':
        from .cnn import DropoutPadBNLeNet5
        nn_obj = DropoutPadBNLeNet5
    elif name == 'MoreChannel':
        from .cnn import MoreChannel
        nn_obj = MoreChannel
    elif name == 'FinalCNN':
        from .cnn import FinalCNN
        nn_obj = FinalCNN
    # ---------------------------------------------------- #
    else:
        raise ValueError(f"Unknown neural network name: {name}")
    return nn_obj(**param)