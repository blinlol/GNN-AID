import warnings
import sys
import logging

from nas.controller import NasController
from nas.search_space import SearchSpace
from nas.trainer import Trainer

warnings.filterwarnings("ignore")

old_stdout = sys.stdout
sys.stdout = open("/home/ubuntu/GNN-AID/src/nas/out", "w")

cora =  ("single-graph", "Planetoid", 'Cora')
mutag = ('multiple-graphs', 'TUDataset', 'MUTAG')

ss = SearchSpace(mutag)
nas = NasController(ss)
trainer = Trainer(ss, nas)
# nas.sample()
trainer.train()

logging.info("best structure: %r", trainer.derive(10))

# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader

# m = TUDataset('/tmp/mutag', 'MUTAG')
# train_mask = torch.BoolTensor([True] * 150 + [False] * 33)
# train_dataset = m.index_select(train_mask)
# train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

