import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import random
import numpy as np


def create_duplicate_dataset(DatasetBaseClass):
    class DupDataset(DatasetBaseClass):

        def __init__(self, copy, **kwargs):
            super(DupDataset, self).__init__(**kwargs)

            self.copy = copy
            self.length = super(DupDataset, self).__len__() # 实际的长度，调用父类（DatasetBaseClass）的 __len__ 方法

        def __len__(self):
            return self.copy * self.length # 复制之后的长度

        def __getitem__(self, index):
            true_index = index % self.length # 将复制后的索引映射回原始索引
            return super(DupDataset, self).__getitem__(true_index)

        def get_img_info(self, index):
            true_index = index % self.length
            return super(DupDataset, self).get_img_info(true_index)

    return DupDataset
