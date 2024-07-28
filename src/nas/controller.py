import logging
import torch
import torch.nn.functional as F

from pydantic import BaseModel, Field

from nas.search_space import SearchSpace, SSType, GNN
from nas.misc import get_variable


logger = logging.getLogger(__name__)


class ControllerArgs(BaseModel):
    class Forward(BaseModel):
        # не понимаю зачем нужны
        mode: str = 'train'
        softmax_temp: float = 5.
        tanh_c: float = 2.5

    
    class Reset(BaseModel):
        # радиус значений в котором инициализировать параметры
        init_range: float = 0.1

    # второе измерение матрицы hidden
    hidden_size: int = 100
    reset: Reset = Field(default_factory=Reset)
    forward: Forward = Field(default_factory=Forward)
    # количество эпох контроллера, после которого выравнивать вероятности методоа
    # (работает только при условии, что SearchSpace.args.type == dynamic_prob)
    dynamic_nas_steps: int = -1


class NasController(torch.nn.Module):
    def __init__(self, search_space: SearchSpace, args: ControllerArgs):
        super().__init__()

        self.ss = search_space
        self.args = args
        self.num_steps = 0

        if self.ss.args.type == SSType.dynamic_prob:
            assert self.args.dynamic_nas_steps > 0

        self.num_tokens = []
        for key, val in self.ss.dict.items():
            self.num_tokens.append(len(val))

        self.encoder = torch.nn.Embedding(
            num_embeddings=sum(self.num_tokens),
            embedding_dim=self.args.hidden_size
        )
        self.lstm = torch.nn.LSTMCell(
            input_size=self.args.hidden_size,
            hidden_size=self.args.hidden_size
        )

        self.decoders = torch.nn.ModuleDict()
        for key, val in search_space.dict.items():
            decoder = torch.nn.Linear(
                in_features=self.args.hidden_size,
                out_features=len(val)
            )
            self.decoders[key] = decoder
        
        self.reset_parameters()

    def reset_parameters(self):
        init_range = self.args.reset.init_range
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            self.decoders[decoder].bias.data.fill_(0)
    
    def forward(self, inputs, hidden, action):
        embed = inputs
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[action](hx) / self.args.forward.softmax_temp
        if self.args.forward.mode == 'train':
            logits = (self.args.forward.tanh_c * torch.tanh(logits))
        return logits, (hx, cx)
    
    def construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.ss.list):
                predicted_actions = self.ss.dict[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def sample(self, batch_size=1, with_details=False):
        '''семплирует batch_size архитектур из пространства поиска переданного конструктором'''
        inputs = torch.zeros([batch_size, self.args.hidden_size])
        hidden = (torch.zeros([batch_size, self.args.hidden_size]), torch.zeros([batch_size, self.args.hidden_size]))
        entropies = []
        log_probs = []
        actions = []

        for action_name in self.ss.list:
            decoder_i = self.ss.ind_by_name(action_name)

            logits, hidden = self.forward(inputs, hidden, action_name)
            if action_name == GNN.str() and \
                    self.ss.args.type == SSType.dynamic_prob and self.num_steps > self.args.dynamic_nas_steps:
                indexes = self.ss.duplicated_gnns_indexes
                for i in indexes:
                    # будет ли это верно математически????
                    # правильно ли???
                    logits[0][i] = 0
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

    # def init_hidden(self, batch_size=64):
    #     # batch_size за что отвечает???
    #     zeros = torch.zeros(batch_size, self.args.hidden_size)
    #     return (get_variable(zeros, requires_grad=False),
    #             get_variable(zeros.clone(), requires_grad=False))
