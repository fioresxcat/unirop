from torch.utils.data import Dataset, DataLoader
from .transforms.roor_ops import *
from .transforms.generic_ops import *
from .transforms.base import ItemTransform
from typing_extensions import List, Dict, Literal, Tuple, Union, Any
import lightning.pytorch as pl


class RORelationDataset(Dataset):
    def __init__(
        self,
        mode: Literal['train', 'val', 'test'],
        data_dirs: List,
        processor_path: str,
        transform_ops: List[ItemTransform],
        augment_ops: List[ItemTransform]
    ):
        super().__init__()
        self.mode = mode
        self.image_paths, self.json_paths = RORelationDataset.prepare_data(data_dirs)
        self.processor_path = processor_path
        self.transform_ops = transform_ops
        self.augment_ops = augment_ops


    @staticmethod
    def prepare_data(data_dirs):
        ipaths, jpaths = [], []
        for data_dir in data_dirs:
            for ip in Path(data_dir).glob('*'):
                if not is_image(ip):
                    continue
                jp = ip.with_suffix('.json')
                with open(jp) as f:
                    list_segments = json.load(f)
                if len(list_segments) == 0:
                    continue
                ipaths.append(ip)
                jpaths.append(jp)
        
        return ipaths, jpaths
    

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        ip, jp = self.image_paths[index], self.json_paths[index]
        item = {'image_path': ip, 'json_path': jp}
        if self.mode == 'train':
            for ops in self.augment_ops:
                item = ops.process(item)

        for ops in self.transform_ops:
            item = ops.process(item, mode=self.mode)
        
        return item
    


class RORelationDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dirs, 
        processor_path: str, 
        augment_ops: List[ItemTransform], 
        transform_ops: List[ItemTransform], 
        batch_size: int, 
        num_workers: int
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.ds_config = {
            'processor_path': processor_path,
            'augment_ops': augment_ops,
            'transform_ops': transform_ops,
        }
        self.num_workers = num_workers
        self.batch_size = batch_size


    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if stage in ['fit', 'validate']:
            self.train_ds = RORelationDataset(
                mode='train',
                data_dirs=self.data_dirs['train'],
                **self.ds_config
            )

            self.val_ds = RORelationDataset(
                mode='val',
                data_dirs=self.data_dirs['val'],
                **self.ds_config
            )

            print(f'NUM TRAIN FILES: {len(self.train_ds)}')
            print(f'NUM VAL FILES: {len(self.val_ds)}')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)