import logging
import warnings
import sys
import datetime as dt
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_file_as, to_yaml_file, to_yaml_str

from nas.controller import NasController, RandomSearchController, ControllerArgs, DynamicBehaviourType
from nas.search_space import SearchSpace, SearchSpaceArgs, SSType
from nas.trainer import Trainer, RandomTrainer, TrainerArgs

logfile = None


class Datasets(Enum):
    cora = ["single-graph", "Planetoid", 'Cora']
    citeceer = ["single-graph", "Planetoid", 'CiteSeer']
    pubmed = ["single-graph", "Planetoid", 'PubMed']

    mutag = ['multiple-graphs', 'TUDataset', 'MUTAG']
    bzr = ['multiple-graphs', 'TUDataset', 'BZR']
    cox2 = ['multiple-graphs', 'TUDataset', 'COX2']
    proteins = ['multiple-graphs', 'TUDataset', 'PROTEINS']
    aids = ['multiple-graphs', 'TUDataset', 'AIDS']


class ExperimentArgs(BaseModel):
    nas_args: ControllerArgs = Field(default_factory=ControllerArgs)
    trainer_args: TrainerArgs = Field(default_factory=TrainerArgs)
    ss_args: SearchSpaceArgs = Field(default_factory=SearchSpaceArgs)
    dataset: Datasets
    derive_num: int = 10
    logfname: str = Field(default_factory=lambda: dt.datetime.now().strftime("%d-%H:%M:%S"))


def set_logfile(fname):
    global logfile

    if logfile:
        logfile.close()
    logfile = open("/home/ubuntu/GNN-AID/src/nas/logs/" + fname, "w")
    logging.basicConfig(
        stream=logfile,
        level='DEBUG',
        force=True
    )


def experiment(cfg_file):
    args = parse_yaml_file_as(ExperimentArgs, cfg_file)
    set_logfile(args.logfname)
    logger.info(to_yaml_str(args))

    ss = SearchSpace(args.dataset.value, args.ss_args)
    nas = NasController(ss, args.nas_args)
    trainer = Trainer(ss, nas, args.trainer_args)

    trainer.train()
    logging.info("%r cfg_file best structure: %r", cfg_file, trainer.derive(args.derive_num))


def random_experiments(cfg_file):
    args = parse_yaml_file_as(ExperimentArgs, cfg_file)
    set_logfile(args.logfname)
    logger.info(to_yaml_str(args))

    ss = SearchSpace(args.dataset.value, args.ss_args)
    nas = RandomSearchController(ss, args.nas_args)
    trainer = RandomTrainer(ss, nas, args.trainer_args)

    trainer.train()
    logging.info("%r cfg_file best structure: %r", cfg_file, trainer.derive(args.derive_num))


def create_random_configs(datasets: list[Datasets], from_i: int, debug: bool=False):
    args = ExperimentArgs(dataset=Datasets.cora)
    args.ss_args.debug = debug
    args.trainer_args.train.num_eras = 30
    args.trainer_args.train.save_eras = 300
    args.trainer_args.train_controller.epochs = 20

    for dataset in datasets:
        args.dataset = dataset
        args.logfname = f"{dataset.value[2].lower()}_random.log"
        to_yaml_file(f"cfg/{from_i}.yml", args)
        from_i += 1


def create_configs(datasets: list[Datasets], from_i: int, debug: bool=False):
    probs_behaviour = [
        "basic",
        "fixed",
        "dynamic-both",
        "dynamic-probsonly",
    ]
    combinations_behaviour = [
        "combinations",
        "no-combinations",
    ]

    args = ExperimentArgs(dataset=Datasets.cora)
    args.ss_args.debug = debug
    args.nas_args.dynamic_nas_steps = 150
    args.trainer_args.train.num_eras = 30
    args.trainer_args.train.save_eras = 30
    args.trainer_args.train_controller.epochs = 20
    args.ss_args.prob_scale = 4

    for dataset in datasets:
        args.dataset = dataset
        for probs in probs_behaviour:
            match probs:
                case "basic":
                    args.nas_args.dynamic_behaviour = DynamicBehaviourType.undefined
                    args.ss_args.type = SSType.basic
                case "fixed":
                    args.nas_args.dynamic_behaviour = DynamicBehaviourType.undefined
                    args.ss_args.type = SSType.fixed_prob
                case "dynamic-both":
                    args.nas_args.dynamic_behaviour = DynamicBehaviourType.both
                    args.ss_args.type = SSType.dynamic_prob
                case "dynamic-probsonly":
                    args.nas_args.dynamic_behaviour = DynamicBehaviourType.probs_only
                    args.ss_args.type = SSType.dynamic_prob
                case _:
                    raise ValueError()
            for comb in combinations_behaviour:
                args.ss_args.with_combinations = comb == "combinations"
                args.logfname = f"{dataset.value[2].lower()}_{probs}_{comb}.log"
                to_yaml_file(f"cfg/{from_i}.yml", args)
                from_i += 1


warnings.filterwarnings("ignore")

old_stdout = sys.stdout
sys.stdout = open("/home/ubuntu/GNN-AID/src/nas/logs/out", "w")

logger = logging.getLogger(__name__)

cfg_dir = "/home/ubuntu/GNN-AID/src/nas/cfg/"

# create_random_configs([Datasets.aids], 34, True)
# random_experiments(cfg_dir + "34.yml")

# cfgs = [
#     cfg_dir + str(i) + '.yml' for i in range(1, 17)
# ]

# for cfg_file in cfgs:
#     try:
#         experiment(cfg_file)
#     except Exception as e:
#         logger.error("cfg_file = %r\n%s", cfg_file, e)

# to_yaml_file("cfg/all.yml", ExperimentArgs(dataset=Datasets.bzr))
# experiment(cfg_dir + "all.yml")