import logging
import numpy as np
import pickle
import torch

from pydantic import BaseModel, Field

from aux.configs import (
    ModelManagerConfig,
    ModelModificationConfig,
    ModelConfig,
    ModelStructureConfig
)
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.gnn_models import FrameworkGNNModelManager, Metric

from nas.misc import get_variable, to_item
from nas.search_space import SearchSpace
from nas.controller import NasController


logger = logging.getLogger(__name__)
history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


class TrainerArgs(BaseModel):
    class Train(BaseModel):
        # количество итераций обучения контроллера
        num_epochs: int = 3
        # периодичность сохранения тренера
        save_epoch: int = 2
        # флаг запуска выполнения derive после обучения
        derive_finaly: bool = False
    
    class Eval(BaseModel):
        # количество эпох обучения модели
        steps: int = 4
        # передается в train_model, хз зачем
        save_model_flag: bool = False
    
    class TrainController(BaseModel):
        # количество эпох обучения контроллера на каждой итерации
        epochs: int = 3
    
    class Derive(BaseModel):
        # нереализованная функциональность
        derive_from_history: bool = False
    
    train: Train = Field(default_factory=Train)
    eval: Eval = Field(default_factory=Eval)
    train_controller: TrainController = Field(default_factory=TrainController)
    derive: Derive = Field(default_factory=Derive)
    # save fname


class Trainer:
    def __init__(self, ss: SearchSpace, nas: NasController, args: TrainerArgs):
        self.ss = ss
        self.nas = nas
        self.args = args

        # move optimizer to contoller?
        self.controller_step = 0  # counter for controller
        self.controller_optim = torch.optim.Adam(self.nas.parameters(), lr=0.005)


    def save(self):
        fname = f"save/Trainer_{self.controller_step}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(self, f)


    def train(self):
        num_epochs = self.args.train.num_epochs
        save_epoch = self.args.train.save_epoch
        derive_finaly = self.args.train.derive_finaly

        for epoch in range(num_epochs):
            self.train_controller()
            # семплирует и измерять архитектуры, не понятно зачем
            # self.derive(derive_num_sample)
            # сохраняет если нужно
            if epoch % save_epoch == 0:
                self.save()
        
        if derive_finaly:
            self.best_structure = self.derive()
            logger.info("Trainer::train self.best_structure = %r", self.best_structure)
        self.save()
        
    def eval(self, sampled_gnn):
        """обучает переданную архитектуру и возвращает скор на валидации
        """
        # TODO: оптимизировать вызовы конструкторов

        steps = self.args.eval.steps
        save_model_flag = self.args.eval.save_model_flag

        manager_config = ModelManagerConfig(**{
                "mask_features": [],
                "optimizer": {
                    "_class_name": "Adam",
                    "_config_kwargs": {},
                }
            }
        )

        gnn = self.ss.create_gnn_structure(sampled_gnn)
        gnn = FrameworkGNNConstructor(
                model_config=ModelConfig(
                    structure=ModelStructureConfig(gnn)
                )
            )
        gnn_model_manager = FrameworkGNNModelManager(
            gnn=gnn,
            dataset_path=self.ss.results_dataset_path,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=steps)
        )
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0

        train_test_split_path = gnn_model_manager.train_model(gen_dataset=self.ss.dataset, steps=steps,
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
        logger.info("Trainer::eval sampled_gnn = %r, metric_loc = %r",sampled_gnn, metric_loc)
        return metric_loc['val']['Accuracy']

    def get_reward(self, sampled_gnns, entropies=None):
        """
        sampled_gnns: list of sampled structures from nas
        entropies: if None return list, else np
        return: list of rewards 
        """
        # TODO: add entropies

        rewards = [] # list of validation accuracies
        for gnn in sampled_gnns:
            rewards.append(self.eval(gnn))

        if entropies is None:
            return rewards
        return rewards * np.ones_like(entropies)

    def train_controller(self):
        # parameters
        controller_epochs = self.args.train_controller.epochs

        # init nas
        self.nas.train() # set training mode
        # hidden = self.nas.init_hidden()

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

    def derive(self, sample_num=None):
        """возвращает лучшую архитектуру"""
        derive_from_history = self.args.derive.derive_from_history

        if sample_num is None and derive_from_history:
            #TODO: поддержать возможность из файла брать историю
            return self.derive_from_history()
        
        sampled_structures = self.nas.sample(sample_num)
        rewards = self.get_reward(sampled_structures)
        min_i = np.argmax(rewards)
        return sampled_structures[min_i]
