import logging
from enum import StrEnum

from collections import OrderedDict
from copy import deepcopy

from base.datasets_processing import DatasetManager
from nas.misc import (
    gcn_layer,
    gin_layer,
    lin_layer,
    connection,
    change_in_out
)


logger = logging.getLogger(__name__)


class GNN(StrEnum):
    gcn = 'gcn'
    gin = 'gin'

    @classmethod
    def str(cls):
        return 'gnn'



class Pool(StrEnum):
    add = 'global_add_pool'
    mean = 'global_mean_pool'

    @classmethod
    def str(cls):
        return 'pool'


dict_layer_by_gnn = {
        GNN.gcn: gcn_layer,
        GNN.gin: gin_layer
    }


class SearchSpace:
    def __init__(self, dataset_full_name):
        self.dataset, self.data, self.results_dataset_path = DatasetManager.get_by_full_name(
            full_name=dataset_full_name,
            dataset_attack_type='original',
            dataset_ver_ind=0
        )
        self.dataset.train_test_split(percent_train_class=0.6)
        self.graph_level = self.dataset.is_multi()

        self.ss = OrderedDict({
            GNN.str(): [g.split('.')[-1] for g in GNN]
        })
        if self.graph_level:
            self.ss[Pool.str()] = [p.split('.')[-1] for p in Pool]

    @property
    def dict(self) -> OrderedDict:
        return self.ss

    @property
    def list(self) -> list[str]:
        if self.graph_level:
            return [GNN.str(), GNN.str(), Pool.str()]
        return [GNN.str(), GNN.str()]
    
    def ind_by_name(self, action_name: str) -> int:
        for i, key in enumerate(self.ss):
            if key == action_name:
                return i    

    def node_task_structure(self, sampled_structure, num_feat, num_classes):
        structure = []
        num_layers = len(sampled_structure[GNN.str()])
        for i, gnn in enumerate(sampled_structure[GNN.str()]):
            layer = deepcopy(dict_layer_by_gnn[gnn])
            in_dim = 16
            out_dim = 16

            if i == 0:
                in_dim = num_feat
            if i == num_layers - 1:
                out_dim = num_classes

            structure.append(
                change_in_out(layer, in_dim, out_dim)
            )
        
        structure[-1]['activation']['activation_name'] = 'LogSoftmax'
        return structure

    def graph_task_structure(self, sampled_structure, num_feat, num_classes):
        structure = []
        num_node_layers = len(sampled_structure[GNN.str()])

        for i, gnn in enumerate(sampled_structure[GNN.str()]):
            layer = deepcopy(dict_layer_by_gnn[gnn])
            in_dim = 16
            out_dim = 16

            if i == 0:
                in_dim = num_feat

            structure.append(
                change_in_out(layer, in_dim, out_dim)
            )

        # graph classifier
        structure.append(change_in_out(deepcopy(lin_layer), out_dim, out_dim))
        structure.append(change_in_out(deepcopy(lin_layer), out_dim, num_classes))

        # connection
        c = deepcopy(connection)
        c['into_layer'] = num_node_layers
        if pool_type := sampled_structure.get('pool'):
            c['connection_kwargs']['pool']['pool_type'] = pool_type
        structure[num_node_layers - 1]['connections'].append(c)

        structure[num_node_layers - 1]['activation']['activation_name'] = 'LogSoftmax'
        structure[-1]['activation']['activation_name'] = 'LogSoftmax'

        return structure

    def create_gnn_structure(self, sampled_gnn):
        """
        sampled_gnn: [gin, gcn, global_add_pool, ...]

        return: dict в формате для передачи сразу в aux.configs.ModelStructureConfig
        """
        # sampled_structure: {gnn: [...], pool: ..., ...}
        num_feat = self.dataset.num_node_features
        num_classes = self.dataset.num_classes
        sampled_structure = {}
 
        for key, val in zip(self.list, sampled_gnn):
            if key == GNN.str():
                sampled_structure[key] = sampled_structure.get(key, []) + [val]
            else:
                sampled_structure[key] = val

        if self.graph_level:
            return self.graph_task_structure(sampled_structure, num_feat, num_classes)
        return self.node_task_structure(sampled_structure, num_feat, num_classes)
