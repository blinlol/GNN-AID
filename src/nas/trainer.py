from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric as pg
from old_code import (
    change_in_out,

    init_cora,
    init_mutag,

    gcn_layer,
    gin_layer,
    lin_layer,
    connection
)

#TODO: use enum for search space keys, vals (pydantic)


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    # if cuda:
    #     out = torch.autograd.Variable(inputs.cuda(), **kwargs)
    # else:
    out = torch.autograd.Variable(inputs, **kwargs)
    return out


class SearchSpace:
    dict_layer_by_gnn = {
        'gcn': gcn_layer,
        'gin': gin_layer
    }

    def __init__(self, graph_level=False):
        self.graph_level = graph_level
        self.ss = OrderedDict({
            'gnn': ['gcn', 'gin']
        })
        if graph_level:
            self.ss['pool'] = ['global_add_pool', 'global_mean_pool']

    
    @property
    def dict(self) -> OrderedDict:
        return self.ss

    @property
    def list(self) -> list[str]:
        if self.graph_level:
            return ['gnn', 'gnn', 'pool']
        return ['gnn', 'gnn']
    
    def ind_by_name(self, action_name: str) -> int:
        for i, key in enumerate(self.ss):
            if key == action_name:
                return i    

    def node_task_structure(self, sampled_structure, num_feat, num_classes):
        structure = []
        num_layers = len(sampled_structure['gnn'])
        for i, gnn in enumerate(sampled_structure['gnn']):
            layer = deepcopy(SearchSpace.dict_layer_by_gnn[gnn])
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
        num_node_layers = len(sampled_structure['gnn'])

        for i, gnn in enumerate(sampled_structure['gnn']):
            layer = deepcopy(SearchSpace.dict_layer_by_gnn[gnn])
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

        structure[num_node_layers - 1]['activation']['activation_name'] = 'LogSoftMax'
        structure[-1]['activation']['activation_name'] = 'LogSoftMax'

        return structure

    def create_gnn_structure(self, sampled_gnn, num_feat, num_classes):
        """
        sampled_gnn: [gin, gcn, global_add_pool, ...]

        return: dict в формате для передачи сразу в aux.configs.ModelStructureConfig
        """
        # sampled_structure: {gnn: [...], pool: ..., ...}
        sampled_structure = {}
        for key, val in zip(self.list, sampled_gnn):
            if key == 'gnn':
                sampled_structure[key] = sampled_structure.get(key, []) + [val]
            else:
                sampled_structure[key] = val

        if self.graph_level:
            return self.graph_task_structure(sampled_structure, num_feat, num_classes)
        return self.node_task_structure(sampled_structure, num_feat, num_classes)



class NasController(torch.nn.Module):
    def __init__(self, search_space: SearchSpace, 
                 controller_hid=100,
                 mode='train'):
        super().__init__()

        self.search_space = search_space
        self.controller_hid = controller_hid
        self.mode = mode
        
        self.num_tokens = []
        for key, val in self.search_space.dict.items():
            self.num_tokens.append(len(val))
        self.encoder = torch.nn.Embedding(
            num_embeddings=sum(self.num_tokens),
            embedding_dim=controller_hid
        )

        self.lstm = torch.nn.LSTMCell(
            input_size=controller_hid,
            hidden_size=controller_hid
        )

        self.decoders = torch.nn.ModuleDict()
        for key, val in search_space.dict.items():
            decoder = torch.nn.Linear(
                in_features=controller_hid,
                out_features=len(val)
            )
            self.decoders[key] = decoder
        
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            self.decoders[decoder].bias.data.fill_(0)
    
    def forward(self, inputs, hidden, action):
        softmax_temp = 5
        tanh_c = 2.5

        embed = inputs
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[action](hx) / softmax_temp
        if self.mode == 'train':
            logits = (tanh_c * torch.tanh(logits))

        return logits, (hx, cx)
    
    def construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.search_space.list):
                predicted_actions = self.search_space.dict[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def sample(self, batch_size=1):
        '''семплирует batch_size архитектур из пространства поиска переданного конструктором'''
        inputs = torch.zeros([batch_size, self.controller_hid])
        hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
        entropies = []
        log_probs = []
        actions = []
        for block_i, action_name in enumerate(self.search_space.list):
            decoder_i = self.search_space.ind_by_name(action_name)

            logits, hidden = self.forward(inputs, hidden, action_name)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = get_variable(
                action[:, 0] + sum(self.num_tokens[:decoder_i]),
                requires_grad=False)

            inputs = self.encoder(inputs)

            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = self.construct_action(actions)
        return dags





c = NasController(SearchSpace())
print(*c.sample(20), sep='\n')
