import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from easydict import EasyDict


class BaseLightningModule(pl.LightningModule):
    def __init__(self, optimizer_config):
        super().__init__()
        self.optimizer_config = EasyDict(optimizer_config)


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.optimizer_config.learning_rate, weight_decay=self.optimizer_config.weight_decay)
        scheduler_config = self.optimizer_config.lr_scheduler

        ckpt_callback = [cb for cb in self.trainer.callbacks if isinstance(cb, ModelCheckpoint)][0]
        
        sched_name = scheduler_config['name']
        if sched_name != 'constant':
            scheduler = torch.optim.lr_scheduler.__dict__[sched_name](
                opt,
                **scheduler_config[sched_name]
            )

            return {
                'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    # 'monitor': ckpt_callback.monitor,
                    'frequency': scheduler_config['frequency'],
                    'interval': scheduler_config['interval']
                }
            }
        else:
            return {
                'optimizer': opt,
            }
    

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch <= int(self.optimizer_config.n_warmup_epochs):
            lr_scale = 0.75 ** (self.optimizer_config.n_warmup_epochs - self.trainer.current_epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.optimizer_config.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def on_train_start(self) -> None:
        if self.optimizer_config.reset_optimizer:
            opt = type(self.trainer.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reset')


    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass