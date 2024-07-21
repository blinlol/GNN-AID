from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric as pg

from aux.configs import ModelManagerConfig, ModelModificationConfig, ModelConfig, ModelStructureConfig
from base.datasets_processing import DatasetManager
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.gnn_models import FrameworkGNNModelManager, Metric


#TODO: use enum for search space keys, vals (pydantic)
gcn_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'GCNConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'aggr': 'add',
            'improved': False,
            'add_self_loops': True,
            'normalize': True,
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
gin_layer = {
    'label': 'n',
    'layer': {
        'gin_seq': [
            {
                'layer': {
                    'layer_name': 'Linear',
                    'layer_kwargs': {
                        'in_features': None,
                        'out_features': 16
                    }
                },
                'batchNorm': {
                    'batchNorm_name': 'BatchNorm1d',
                    'batchNorm_kwargs': {
                        'momentum': 0.1,
                        'affine': True,
                        'num_features': 16,
                        'eps': 1e-05
                    }
                },
                'activation': {
                    'activation_name': 'ReLU',
                    'activation_kwargs': {}
                }
            },
            {
                'layer': {
                    'layer_name': 'Linear',
                    'layer_kwargs': {
                        'in_features': 16,
                        'out_features': None
                    }
                },
                'activation': {
                    'activation_name': 'ReLU',
                    'activation_kwargs': {}
                }
            }
        ],
        'layer_name': 'GINConv',
        'layer_kwargs': {}
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
connection = {
    'into_layer': None,  # index of the last layer
    'connection_kwargs': {
        'pool': {
            'pool_type': 'global_add_pool'
        },
        'aggregation_type': 'cat'
    }
}
lin_layer = {
    'label': 'g',
    'layer': {
        'layer_name': 'Linear',
        'layer_kwargs': {
            'in_features': 48,
            'out_features': None,
        },
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None,
    },
}


def change_in_out(layer: dict, in_features: int=None, out_features: int=None):
    if layer['layer']['layer_name'] == 'GINConv':
        return change_in_out_gin(layer, in_features, out_features)

    layer = deepcopy(layer)
    if 'in_channels' in layer['layer']['layer_kwargs']:
        if in_features is not None:
            layer['layer']['layer_kwargs']['in_channels'] = in_features
        if out_features is not None:
            layer['layer']['layer_kwargs']['out_channels'] = out_features
    elif 'in_features' in layer['layer']['layer_kwargs']:
        if in_features is not None:
            layer['layer']['layer_kwargs']['in_features'] = in_features
        if out_features is not None:
            layer['layer']['layer_kwargs']['out_features'] = out_features
    else:
        raise ValueError('cant change in/out features')
    return layer


def change_in_out_gin(layer: dict, in_features: int=None, out_features: int=None):
    layer = deepcopy(layer)
    if in_features is not None:
        layer['layer']['gin_seq'][0]['layer']['layer_kwargs']['in_features'] = in_features
    if out_features is not None:
        layer['layer']['gin_seq'][-1]['layer']['layer_kwargs']['out_features'] = out_features
    return layer


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    # if cuda:
    #     out = torch.autograd.Variable(inputs.cuda(), **kwargs)
    # else:
    out = torch.autograd.Variable(inputs, **kwargs)
    return out


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def get_dataset(full_name):
    return DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_attack_type='original',
        dataset_ver_ind=0
    )


history = []

def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value




class SearchSpace:
    dict_layer_by_gnn = {
        'gcn': gcn_layer,
        'gin': gin_layer
    }

    def __init__(self, dataset_full_name, graph_level=False):
        # self.graph_level = graph_level
        self.dataset, self.data, self.results_dataset_path = DatasetManager.get_by_full_name(
            full_name=dataset_full_name,
            dataset_attack_type='original',
            dataset_ver_ind=0
        )
        self.dataset.train_test_split()
        self.graph_level = dataset_full_name[0] == 'multiple-graphs'

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

    def sample(self, batch_size=1, with_details=False):
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

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)
        return dags

    def init_hidden(self, batch_size=64):
        # batch_size за что отвечает???
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (get_variable(zeros, requires_grad=False),
                get_variable(zeros.clone(), requires_grad=False))


class Trainer:
    def __init__(self, ss: SearchSpace, nas: NasController):
        self.ss = ss
        self.nas = nas

        # move optimizer to contoller?
        self.controller_step = 0  # counter for controller
        self.controller_optim = torch.optim.Adam(self.nas.parameters(), lr=0.005)

    
    def train(self):
        num_epochs = 3

        for epoch in range(num_epochs):
            self.train_controller()
            self.derive()
    
    def get_reward(self, sampled_gnns, entropies):
        """
        sampled_gnns: list of sampled structures from nas
        """


        manager_config = ModelManagerConfig(**{
                "mask_features": [],
                "optimizer": {
                    "_class_name": "Adam",
                    "_config_kwargs": {},
                }
            }
        )
        steps_epochs = 1
        save_model_flag = True
        rewards = [] # list of validation accuracies
        for gnn in sampled_gnns:
            gnn = self.ss.create_gnn_structure(gnn)
            gnn = FrameworkGNNConstructor(
                    model_config=ModelConfig(
                        structure=ModelStructureConfig(gnn)
                    )
                )
            gnn_model_manager = FrameworkGNNModelManager(
                gnn=gnn,
                dataset_path=self.ss.results_dataset_path,
                manager_config=manager_config,
                modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
            )
            gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0

            train_test_split_path = gnn_model_manager.train_model(gen_dataset=self.ss.dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("Accuracy", mask='train')])

            if train_test_split_path is not None:
                self.ss.dataset.save_train_test_mask(train_test_split_path)
                train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                    :]
                self.ss.dataset.train_mask, self.ss.dataset.val_mask, self.ss.dataset.test_mask = train_mask, val_mask, test_mask
                self.ss.data.percent_train_class, self.ss.data.percent_test_class = train_test_sizes


            metric = Metric("Accuracy", mask='val')
            metric_loc = gnn_model_manager.evaluate_model(
                gen_dataset=self.ss.dataset, metrics=[metric])
            print(metric_loc)
            # TODO: add entropies
            rewards.append(metric_loc['val']['Accuracy'])
        rewards = rewards * np.ones_like(entropies)
        return rewards

    def train_controller(self):
        # parameters
        controller_epochs = 3

        # init nas
        self.nas.train() # set training mode
        hidden = self.nas.init_hidden()

        # definitions
        total_loss = 0
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        for step in range(controller_epochs):
            structures, log_probs, entropies = self.nas.sample(with_details=True)
            # обучить модели, получить метрику (возможно внести небольшой шум в награду)
            np_entropies = entropies.data.cpu().numpy()
            rewards = self.get_reward(structures, 
                                      np_entropies)
            # сравнить аналогично с топом, прокидываем лосс

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            if baseline is None:
                baseline = rewards
            else:
                decay = 0.95
                baseline = decay * baseline + (1 - decay) * rewards
            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = get_variable(adv, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()

            total_loss += to_item(loss.data)

            self.controller_step += 1


    def derive(self):
        pass


# def train_controller():


if __name__ == '__main__':
    cora =  ("single-graph", "Planetoid", 'Cora')
    mutag = ('multiple-graphs', 'TUDataset', 'MUTAG')

    ss = SearchSpace(cora)
    nas = NasController(ss)
    trainer = Trainer(ss, nas)
    nas.sample()
    trainer.train()

    # from torch_geometric.datasets import TUDataset
    # from torch_geometric.data import DataLoader

    # m = TUDataset('/tmp/mutag', 'MUTAG')
    # train_mask = torch.BoolTensor([True] * 150 + [False] * 33)
    # train_dataset = m.index_select(train_mask)
    # train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
   