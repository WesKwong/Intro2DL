def get_model_obj(hp):
    # no model specified
    if 'model' not in hp:
        from .base_model import BaseModel
        return BaseModel
    # model specified
    name = hp['model']
    if name == 'Lab1':
        from .lab1 import Lab1Model
        return Lab1Model
    else:
        raise ValueError(f"Invalid model: {name}")