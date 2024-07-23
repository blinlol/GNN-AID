import logging
import torch
import torch.nn.functional as F

from nas.search_space import SearchSpace
from nas.misc import get_variable


logger = logging.getLogger(__name__)


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
