def get_model_obj(hp):
    # no model specified
    if 'model' not in hp:
        from .base_model import BaseModel
        return BaseModel
    # model specified
    name = hp['model']
    if name == 'Lab1':
        from .lab1_model import Lab1Model
        return Lab1Model
    elif name == 'GCNModel':
        from .gcn_model import GCNModel
        return GCNModel
    else:
        raise ValueError(f"Invalid model: {name}")