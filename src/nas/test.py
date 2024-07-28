import logging
import warnings
import sys
import datetime as dt
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

from nas.controller import NasController, ControllerArgs
from nas.search_space import SearchSpace, SearchSpaceArgs
from nas.trainer import Trainer, TrainerArgs

logfile = None


class Datasets(Enum):
    cora = ["single-graph", "Planetoid", 'Cora']
    mutag = ['multiple-graphs', 'TUDataset', 'MUTAG']
    bzr = ['multiple-graphs', 'TUDataset', 'BZR']


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

    ss = SearchSpace(args.dataset.value, args.ss_args)
    nas = NasController(ss, args.nas_args)
    trainer = Trainer(ss, nas, args.trainer_args)

    trainer.train()
    logging.info("%r cfg_file best structure: %r", cfg_file, trainer.derive(args.derive_num))



warnings.filterwarnings("ignore")

old_stdout = sys.stdout
sys.stdout = open("/home/ubuntu/GNN-AID/src/nas/logs/out", "w")

cfg_dir = "/home/ubuntu/GNN-AID/src/nas/cfg/"
cfgs = [
    cfg_dir + '1.yml',
    cfg_dir + '2.yml'
]
for cfg_file in cfgs:
    experiment(cfg_file)

# to_yaml_file("cfg/all.yml", ExperimentArgs(dataset=Datasets.bzr))