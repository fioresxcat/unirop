import json
import pdb
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class ItemTransform:
    def __init__(self, keep_keys):
        self.keys = keep_keys

    def keep_keys(self, item):
        if len(self.keys) == 0: # keep all keys
            return item
        
        keys = list(item.keys())
        for k in keys:
            if k not in self.keys:
                item.pop(k)
        return item

    def process(self, item, mode='val'):
        raise NotImplementedError('Method must be implemented')

    def __repr__(self):
        attributes = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"


class BatchItemTransform:
    def __init__(self, keep_keys):
        self.keys = keep_keys

    def keep_keys(self, items):
        for item in items:
            keys = list(item.keys())
            for k in keys:
                if k not in self.keys:
                    item.pop(k)
        return items
    

    def process(self, items):
        raise NotImplementedError('Method must be implemented')


    def __repr__(self):
        attributes = ', '.join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"