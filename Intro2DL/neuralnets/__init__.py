def get_nn(hp):
    name = hp['net']
    if name == "MLP":
        from .mlp import MLP
        return MLP()
    elif name == "CNN":
        from .cnn import CNN
        return CNN()
    else:
        raise ValueError(f"Unknown neural network name: {name}")