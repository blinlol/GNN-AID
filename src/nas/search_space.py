import logging
from enum import StrEnum, IntEnum
from pydantic import BaseModel

from collections import OrderedDict
from copy import deepcopy

from base.datasets_processing import DatasetManager
from nas.misc import (
    gcn_layer, gin_layer, sg_layer, gat_layer,
    gmm_layer, ssg_layer, tag_layer, sage_layer,
    connection, lin_layer,
    change_in_out
)


logger = logging.getLogger(__name__)


class GNN(StrEnum):
    gcn = 'gcn'
    gin = 'gin'
    sg = 'sg'
    ssg = 'ssg'
    gat = 'gat'
    gmm = 'gmm'
    tag = 'tag'
    sage = 'sage'

    @classmethod
    def str(cls):
        return 'gnn'


dict_layer_by_gnn = {
    GNN.gcn: gcn_layer,
    GNN.gin: gin_layer,
    GNN.gat: gat_layer,
    GNN.sg: sg_layer,
    GNN.ssg: ssg_layer,
    GNN.gmm: gmm_layer,
    GNN.tag: tag_layer,
    GNN.sage: sage_layer,
}


class Pool(StrEnum):
    add = 'global_add_pool'
    mean = 'global_mean_pool'
    max = 'global_max_pool'

    @classmethod
    def str(cls):
        return 'pool'


class TrainEpochs(IntEnum):
    @classmethod
    def str(cls):
        return "train_epochs"


class SSType(StrEnum):
    # equal probabilities for all gnns
    basic = 'basic'
    # фиксированная повышенная вероятность для целевых методов
    fixed_prob = 'fixed_prob'
    # со временем вероятность выравнивается
    dynamic_prob = 'dynamic_prob'


class SearchSpaceArgs(BaseModel):
    type: SSType = SSType.basic
    # во сколько раз увеличивать вероятность выпадения целевых методов
    prob_scale: int = 1
    debug: bool = False


class SearchSpace:
    def __init__(self, dataset_full_name, args: SearchSpaceArgs):
        self.dataset, self.data, self.results_dataset_path = DatasetManager.get_by_full_name(
            full_name=dataset_full_name,
            dataset_attack_type='original',
            dataset_ver_ind=0
        )
        self.dataset.train_test_split(percent_train_class=0.6)
        self.graph_level = self.dataset.is_multi()
        self.args = args

        self.main_gnns_indexes = []
        # Индексы в self.ss['gnn'], которые соответствуют повторам методов
        self.duplicated_gnns_indexes = []

        if self.graph_level:
            self.ss = OrderedDict({
                GNN.str(): [g.value for g in GNN],
                Pool.str(): [p.value for p in Pool],
                TrainEpochs.str(): list(range(3000, 8001, 500))
            })
            self.main_gnns_indexes.extend(range(len(self.ss[GNN.str()])))
            if self.args.type in (SSType.fixed_prob, SSType.dynamic_prob):
                scale = self.args.prob_scale - 1
                self.ss[GNN.str()].extend([GNN.gcn.value] * scale)
                self.ss[GNN.str()].extend([GNN.gin.value] * scale)

                l = len(self.ss[GNN.str()])
                self.duplicated_gnns_indexes.extend(range(l - 2 * scale, l))
        else:
            self.ss = OrderedDict({
                GNN.str(): [g.value for g in GNN],
                TrainEpochs.str(): list(range(100, 221, 20))
            })
            self.main_gnns_indexes.extend(range(len(self.ss[GNN.str()])))
            if self.args.type in (SSType.fixed_prob, SSType.dynamic_prob):
                scale = self.args.prob_scale - 1
                self.ss[GNN.str()].extend([GNN.gcn.value] * scale)
                self.ss[GNN.str()].extend([GNN.gmm.value] * scale)

                l = len(self.ss[GNN.str()])
                self.duplicated_gnns_indexes.extend(range(l - 2 * scale, l))
        
        if self.args.debug:
            self.ss[TrainEpochs.str()] = [1]

    @property
    def dict(self) -> OrderedDict:
        return self.ss

    @property
    def list(self) -> list[str]:
        if self.graph_level:
            return [GNN.str(), GNN.str(), Pool.str(), TrainEpochs.str()]
        return [GNN.str(), GNN.str(), TrainEpochs.str()]

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

    def get_train_epochs(self, sampled_gnn):
        return sampled_gnn[self.list.index(TrainEpochs.str())]
