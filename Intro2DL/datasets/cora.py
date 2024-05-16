from .graph_dataset import GraphDataset


class Cora(GraphDataset):

    def __init__(self, path):
        self._name = 'Cora'
        super().__init__(path)
