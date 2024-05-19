from .graph_dataset import GraphDataset


class Citeseer(GraphDataset):

    def __init__(self, path):
        self._name = 'Citeseer'
        super().__init__(path)