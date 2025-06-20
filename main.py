import os
os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'

import json
import torch
import yaml
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch import Trainer
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.cli import LightningCLI
from models.layoutlmv3.layoutlmv3_gp import LayoutLMv3ForRORelationModule
from models.layoutlmv3.layoutlmv3_gp_two_heads import LayoutLMv3TwoHeadsForRORelationModule
from models.layoutlmv3.layoutlmv3_token_cls import LayoutLMv3ForGroupTokenClassificationModule
from dataset.roor import RORelationDataModule
from dataset.token_cls import ROTokenClassificationDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # parser.add_argument('--save_config_callback', default=None)
        parser.add_argument('--save_config_overwrite', default=False)
        parser.add_argument('--common', default={})

    def before_instantiate_classes(self) -> None:
        self.save_config_kwargs['overwrite'] = self.config[self.config.subcommand].save_config_overwrite
        # if self.config[self.config.subcommand].save_config_callback is None:
        #     self.save_config_callback = None

def cli_main():
    cli = MyLightningCLI(
        LayoutLMv3ForRORelationModule,
        # LayoutLMv3TwoHeadsForRORelationModule,
        # LayoutLMv3ForGroupTokenClassificationModule,
        RORelationDataModule,
        # ROTokenClassificationDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
    )



if __name__ == '__main__':
    cli_main()