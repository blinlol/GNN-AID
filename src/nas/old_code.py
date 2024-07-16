# from models_builder.conducting_series_experiments_gnn import ConductingSeriesExperimentsGNN


from aux.configs import ModelManagerConfig, ModelModificationConfig, ModelConfig, ModelStructureConfig
from base.datasets_processing import DatasetManager
from models_builder.gnn_models import Metric, FrameworkGNNModelManager
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.models_zoo import model_configs_zoo

import json
import pandas as pd
from copy import deepcopy
from time import time

def change_k(layer: dict, k: int):
    layer = deepcopy(layer)
    if k == None:
        return layer

    if 'K' in layer['layer']['layer_kwargs']:
        # in SG, SSG, TAG
        layer['layer']['layer_kwargs']['K'] = k
    elif 'heads' in layer['layer']['layer_kwargs']:
        # in GAT
        layer['layer']['layer_kwargs']['heads'] = k

    return layer

def change_aggr(layer: dict, aggr: str):
    # layer = deepcopy(layer)
    # layer['layer']['layer_kwargs']['aggr'] = aggr
    # return layer
    return deepcopy(layer)

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


def prepare_metrics(metrics_val_by_iters: dict, analyze_mode='average',
                          which_ind_return_for_remove='not_use'):
    union_metrics_val = {}
    for ind, val in metrics_val_by_iters.items():
        for mask, metrics in val.items():
            if mask not in union_metrics_val:
                union_metrics_val[mask] = {}
            for metric, metric_val in metrics.items():
                if metric not in union_metrics_val[mask]:
                    union_metrics_val[mask][metric] = [metric_val]
                else:
                    union_metrics_val[mask][metric].append(metric_val)
    # print(json.dumps(union_metrics_val, indent=2))
    return union_metrics_val


def save_experiment_results(metrics: dict,
                            dataset_name: str,
                            layer_name: str,
                            layers_number: int,
                            epochs_number: int,
                            K: int,
                            aggr: str,
                            fname: str = 'results.csv'):
    df = pd.read_csv('testing/Balabanov_test/'+fname, index_col='index')
    for tr_acc, tr_F1, te_acc, te_F1, duration in zip(metrics['train']['Accuracy'], 
                                            metrics['train']['F1'],
                                            metrics['test']['Accuracy'],
                                            metrics['test']['F1'],
                                            metrics['time']['time']):
        df.loc[df.index.max() + 1] = [dataset_name, layer_name, K, aggr, layers_number, epochs_number, duration, tr_acc, tr_F1, te_acc, te_F1]
    df.to_csv('testing/Balabanov_test/'+fname)


def run_experiment(fname: str='results.csv'):
    gnn_mm_class_kwargs.update({'manager_config': ModelManagerConfig(**gnn_mm_class_kwargs['manager_config'])})
    
    gnn_model_manager = gnn_mm_class(**gnn_mm_class_kwargs)
    dir_with_all_ver_gmm = gnn_model_manager.model_path_info().parent
    
    # start_time = time()
    metrics_val_by_iters, ver_indexes_save_info = ConductingSeriesExperimentsGNN.n_training(
        gnn_mm_class=gnn_mm_class, gnn_mm_class_kwargs=gnn_mm_class_kwargs,
        train_kwargs=train_kwargs, evaluate_model_kwargs=evaluate_model_kwargs,
        gen_dataset=dataset,
        use_pretrained_models=use_pretrained_models, models_vers_for_load=models_vers_for_load,
        number_runs=number_runs, models_vers=models_vers,
    )
    # exp_mean_duration = (time() - start_time) / number_runs

    # metrics_without_time = remove_time_from_metrics(metrics_val_by_iters)

    union_metrics_val, models_ind_for_remove = ConductingSeriesExperimentsGNN.models_metric_analyze(
        metrics_val_by_iters=metrics_val_by_iters, analyze_mode=analyze_mode,
        which_ind_return_for_remove=which_ind_return_for_remove,
    )
    
    prepared_metrics = prepare_metrics(metrics_val_by_iters=metrics_val_by_iters, analyze_mode=analyze_mode,
        which_ind_return_for_remove=which_ind_return_for_remove,)
    
    # in SG, SSG, TAG
    K = gnn.structure.layers[0]['layer']['layer_kwargs'].get('K', None)
    if K is None:
        # in GAT
        K = gnn.structure.layers[0]['layer']['layer_kwargs'].get('heads', None)

    # in SAGE, GCN
    aggr = gnn.structure.layers[0]['layer']['layer_kwargs'].get('aggr', None)
    if aggr is None:
        # in GAT
        aggr = gnn.structure.layers[0]['layer']['layer_kwargs'].get('fill_value', None)

    save_experiment_results(prepared_metrics, 
                            full_name[-1], 
                            gnn.structure.layers[0]['layer']['layer_name'],
                            len(gnn.structure.layers), 
                            steps_epochs,
                            K,
                            aggr,
                            fname=fname)
    
    ConductingSeriesExperimentsGNN.remove_models_by_ind(
        dir_with_all_ver_gmm=dir_with_all_ver_gmm,
        models_ind_for_remove=models_ind_for_remove,
        removable_indexes=ver_indexes_save_info
    )

def experiment(gnn_structure, epochs):
    global gnn_mm_class_kwargs, train_kwargs, evaluate_model_kwargs, gnn, steps_epochs

    steps_epochs = epochs
    gnn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                gnn_structure
            )
        )
    )
    gnn_mm_class_kwargs = {'gnn': gnn,
                           'dataset_path': results_dataset_path,
                           'manager_config': manager_config,
                           'modification': ModelModificationConfig()
                           }
    train_kwargs = {
        'gen_dataset': dataset, 
        'steps': epochs,
        'save_model_flag': save_model_flag,
        'metrics': train_metrics
    }
    evaluate_model_kwargs = {
        'gen_dataset': dataset,
        'metrics': evaluate_metrics,
    }
    
    run_experiment()
    
def init_dataset(full_name):
    global dataset, data, results_dataset_path

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_attack_type='original',
        dataset_ver_ind=0
    )

def init_cora():
    global full_name
    full_name = ("single-graph", "Planetoid", 'Cora')
    init_dataset(full_name)    
    
def init_citeseer():
    global full_name
    full_name = ("single-graph", "Planetoid", 'CiteSeer')
    init_dataset(full_name)    

def init_pubmed():
    global full_name
    full_name = ("single-graph", "Planetoid", 'PubMed')
    init_dataset(full_name)

def init_mutag():
    global full_name
    full_name = ('multiple-graphs', 'TUDataset', 'MUTAG')
    init_dataset(full_name)

def init_proteins():
    global full_name
    full_name = ('multiple-graphs', 'TUDataset', 'PROTEINS')
    init_dataset(full_name)

def init_cox2():
    global full_name
    full_name = ('multiple-graphs', 'TUDataset', 'COX2')
    init_dataset(full_name)

def init_bzr():
    global full_name
    full_name = ('multiple-graphs', 'TUDataset', 'BZR')
    init_dataset(full_name)

def init_aids():
    global full_name
    full_name = ('multiple-graphs', 'TUDataset', 'AIDS')
    init_dataset(full_name)

def planetoid_structure(layer, layers_number, k=None, aggr=None, layer_out=16):
    layer = change_aggr(change_k(layer, k), aggr)
    
    structure = []
    
    for i in  range(layers_number):
        if i == 0:
            structure.append(change_in_out(layer, dataset.num_node_features, layer_out))
        elif i == layers_number - 1:
            structure.append(change_in_out(layer, layer_out, dataset.num_classes))
        else:
            structure.append(change_in_out(layer, layer_out, layer_out) )

    structure[-1]['activation']['activation_name'] = 'LogSoftmax'

    if layer['layer']['layer_name'] == 'GATConv' and k is not None and k > 1:
        structure[-1] = change_k(structure[-1], 1)
        
        dim = layer_out * k
        for i in range(1, layers_number - 1):
            structure[i] = change_in_out(structure[i], dim, dim)
            dim *= k
        structure[-1] = change_in_out(structure[-1], dim, dataset.num_classes)
    
    return structure


def tudataset_structure(layer, layers_number, k=None, aggr=None, layer_out=16):
    layer = change_k(layer, k)

    structure = []
    
    for i in range(layers_number):
        if i == 0:
            structure.append(change_in_out(layer, dataset.num_node_features, layer_out))
        else:
            structure.append(change_in_out(layer, layer_out, layer_out))

    structure.append(change_in_out(lin_layer, layer_out, dataset.num_classes))

    if layer['layer']['layer_name'] == 'GATConv' and k is not None and k > 1:
        dim = layer_out * k
        for i in range(1, layers_number):
            structure[i] = change_in_out(structure[i], dim, dim)
            dim *= k
        structure[-1] = change_in_out(structure[-1], dim)
    
    connection_tmp = deepcopy(connection)
    connection_tmp['into_layer'] = len(structure) - 1
    structure[-2]['connections'].append(connection_tmp)

    structure[-2]['activation']['activation_name'] = 'LogSoftmax'
    structure[-1]['activation']['activation_name'] = 'LogSoftmax'

    return structure


def run_experiments_with_params(dataset: str, params: list, structure_fname: str=None):
    if structure_fname:
        structures_f = open(structure_fname, 'a')
    dataset = dataset.lower()

    if dataset in ['cora', 'pubmed', 'citeseer']:
        s = planetoid_structure
    elif dataset in ['mutag', 'proteins', 'cox2', 'bzr', 'aids']:
        s = tudataset_structure
    else:
        raise NotImplementedError('Dataset {} is not supported'.format(dataset))

    if dataset == 'cora':
        init_cora()
    elif dataset == 'citeseer':
        init_citeseer()
    elif dataset == 'pubmed':
        init_pubmed()
    elif dataset == 'proteins':
        init_proteins()
    elif dataset == 'cox2':
        init_cox2()
    elif dataset == 'bzr':
        init_bzr()
    elif dataset == 'aids':
        init_aids()
    elif dataset == 'mutag':
        init_mutag()
    else:
        raise NotImplementedError('Dataset {} is not supported'.format(dataset))

    for p in params:
        experiment(s(**p['structure_kwargs']), p['epochs'])
        if structure_fname:
            print(json.dumps(gnn.structure, indent=4), file=structures_f, end='\n\n')












number_runs = 20
optimizer = 'NAdam'
train_metrics = [Metric("F1", mask='train', average=None)]
evaluate_metrics = [Metric("Accuracy", mask='test'), 
                    Metric("F1", mask='test', average='macro'),
                    Metric("Accuracy", mask='train'), 
                    Metric("F1", mask='train', average='macro')]
save_model_flag = True
use_pretrained_models = True
models_vers_for_load = None
models_vers = None
analyze_mode = 'average'
which_ind_return_for_remove = 'none'
layer_out = 16

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
sage_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'SAGEConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'aggr': 'mean',
            'normalize': False,
            'root_weight': True,
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
gat_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'GATConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'heads': 1,
            'concat': True,
            'negative_slope': 0.2,
            'dropout': 0,
            'add_self_loops': True,
            'edge_dim': None,
            'fill_value': 'mean',
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
sg_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'SGConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'K': 1,
            'cashed': True,
            'add_self_loops': True,
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
ssg_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'SSGConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'alpha': 0.1,
            'K': 1,
            'cashed': True,
            'add_self_loops': True,
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
tag_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'TAGConv',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'K': 3,
            'normalize': False,
            'bias': True
        }
    },
    'activation': {
        'activation_name': 'ReLU',
        'activation_kwargs': None
    },
    'connections': []
}
gmm_layer = {
    'label': 'n',
    'layer': {
        'layer_name': 'GMM',
        'layer_kwargs': {
            'in_channels': None,
            'out_channels': 16,
            'dim': 1,
            'kernel_size': 1,
            'cached': False,
            'separate_gaussians': False,
            'aggr': 'mean',
            'root_weight': True,
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





gnn_mm_class = FrameworkGNNModelManager
manager_config = {
    "optimizer": {"class_name":optimizer},
    'mask_features': [],
    'batch': 1000,
}



#region pubmed

# pubmed_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 320},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 180},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 150},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 250},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 250},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 180},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 400},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 275},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 6}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 7}, 'epochs': },

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 180},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 6}, 'epochs': },
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 7}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 250},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 250},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 500},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 250},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2}, 'epochs': },

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 300},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 300},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 300}
# ]
# number_runs = 5
# run_experiments_with_params('pubmed', pubmed_params)
#endregion






#region citeseer

# citeseer_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 110},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 110},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 110},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 40},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 60},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 70},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 100},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 70},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': 50},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': 50},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': 40},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 5}, 'epochs': 40},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 6}, 'epochs': 40},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 7}, 'epochs': 40},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 80},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 2}, 'epochs': 50},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 3}, 'epochs': 40},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 4}, 'epochs': 40},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 5}, 'epochs': 35},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 6}, 'epochs': 35},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 7}, 'epochs': 35},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 80},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 60},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 50},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 50},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 120},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 80},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 80},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 60},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 100},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 150},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 150},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 150},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 60},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 80},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 100},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 90},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 100},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 130},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 150},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 110},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 120},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 130},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 140},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 120},

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1}, 'epochs': 40},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2}, 'epochs': 100},

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 80},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 170},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 150}
# ]
# number_runs = 5
# run_experiments_with_params('citeseer', citeseer_params)

#endregion



#region cora

# cora_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 170},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 170},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 170},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 100},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 120},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 130},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 140},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 110},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': 80},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': 55},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': 55},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 5}, 'epochs': 55},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 6}, 'epochs': 50},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 7}, 'epochs': 60},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 110},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 2}, 'epochs': 75},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 3}, 'epochs': 75},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 4}, 'epochs': 60},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 5}, 'epochs': 60},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 6}, 'epochs': 50},
#     # {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 7}, 'epochs': 40},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 150},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 160},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 160},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 180},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 180},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 200},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 150},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 150},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 210},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 240},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 240},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 250},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 250},

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1}, 'epochs': 40},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2}, 'epochs': 100},

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 160},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 140},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 120}
# ]
# number_runs = 5
# run_experiments_with_params('cora', cora_params)

#endregion


#region mutag

# mutag_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 7000},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 4000},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1500},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 4500},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 4500},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 4500},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 4500},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 6500},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 11000},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 12000},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 9000},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 2500},


#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 6500},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 7500},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 8000},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 7000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 7000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 7000},

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 4000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 6000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 8000},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 10000},


#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': 4000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': 4000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1500},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': 1000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': 600},

#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': 600},

#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1100},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': 3000},


#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 2500},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 2500},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 2500},


#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 2500},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 2500},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 2500},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': 4500},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': 5000},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': 5000},
# ]
# number_runs = 5
# run_experiments_with_params('mutag', mutag_params)

#endregion


#region proteins

# proteins_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 800},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 1400},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1000},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 800},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 2000},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 2000},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 800},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1600},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1200},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': },

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 1800},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 3500},

#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 1400},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 1200},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 500},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 2000},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 2000},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 2000},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 1000},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 1500},
# ]
# number_runs = 5
# run_experiments_with_params('proteins', proteins_params)

#endregion



#region cox2

# cox2_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 1300},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 1800},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1300},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1600},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1200},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 200},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1400},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1200},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': 500},

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 200},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 400},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 1400},

#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 700},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 1000},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 150},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 200},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1400},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 800},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 200},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 1400},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 1000},
# ]
# number_runs = 5
# run_experiments_with_params('cox2', cox2_params)

#endregion



#region bzr

# bzr_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 400},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 400},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 2500},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1500},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1100},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 300},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1400},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1000},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': 900},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': 500},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': 700},

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 200},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 700},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 1000},

#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 1200},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 1200},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 200},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 2000},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 2200},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1200},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 1500},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 1200},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 2000},
# ]
# number_runs = 5
# run_experiments_with_params('bzr', bzr_params)

#endregion



#region aids

# aids_params = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 1400},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 2000},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1500},

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1400},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 2000},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 2000},
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 400},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 800},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 600},
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': },


#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1000},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1300},
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': },
#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': },

#     # {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': },
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1500},

#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 500},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 600},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 700},

#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 1100},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 900},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 500},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 800},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 600},

#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 450},

#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 400},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 1600},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 1400},
# ]
# number_runs = 5
# run_experiments_with_params('aids', aids_params)

#endregion






# params_one_epoch_cora = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sage_layer, 'layers_number': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 5}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 6}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 7}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 5}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 6}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 3, 'k': 7}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 1}
# ]
# number_runs = 1
# run_experiments_with_params('cora', params_one_epoch_cora, '/home/ubuntu/cora-1-epoch-structure.txt')


# params_one_epoch_mutag = [
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gcn_layer, 'layers_number': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 1, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': sksg_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': ssg_layer, 'layers_number': 3, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 5}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 1, 'k': 6}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': tag_layer, 'layers_number': 3, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gmm_layer, 'layers_number': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 1}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gin_layer, 'layers_number': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 2}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 3}, 'epochs': 1},
#     {'structure_kwargs':{'layer': gat_layer, 'layers_number': 2, 'k': 4}, 'epochs': 1},
# ]
# number_runs = 1
# run_experiments_with_params('mutag', params_one_epoch_mutag, '/home/ubuntu/mutag-1-epoch-structure.txt')
