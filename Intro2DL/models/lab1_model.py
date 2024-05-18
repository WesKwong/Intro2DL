from .base_model import *


class Lab1Model(BaseModel):

    def __init__(self, train_loader, val_loader, hyperparameters, experiment):
        super().__init__(train_loader, val_loader, hyperparameters, experiment)

    def get_nn(self):
        # Neural Network
        data, label = next(iter(self.val_loader))
        input_size = data.shape[1]
        output_size = label.shape[1]
        hidden_sizes = self.hp['hidden_sizes']
        activation = self.hp['activation']
        net_hp = {'name': self.hp['net'],
                  'param': {'input_size': input_size,
                            'output_size': output_size,
                            'hidden_sizes': hidden_sizes,
                            'activation': activation}}
        hp = {'net': net_hp}
        self.net = get_nn(hp).to(device)