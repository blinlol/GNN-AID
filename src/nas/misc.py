import torch
import numpy as np

from copy import deepcopy



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

connection = {
    'into_layer': None,  # index of the last layer
    'connection_kwargs': {
        'pool': {
            'pool_type': 'global_add_pool'
        },
        'aggregation_type': 'cat'
    }
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
