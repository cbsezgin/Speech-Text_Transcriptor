from random import random
from torch.utils.data import DistributedSampler, DataLoader
from dataset import LibriSpeechDataset
from utils import custom_collate_fn, TextTransformer, create_model

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)

class Trainer:
    def __init__(self, config, rank, world_size, from_checkpoint):
        self.device = rank
        self.world_size = world_size

        self.batch_size = config['batch_size']
        self.checkpoint_dir = config['checkpoint_dir']
        self.start_epoch = 1
        self.epochs = config['epochs'] + 1
        self.use_onecyclelr = config['use_onecyclelr']
        self.num_workers = config['num_workers']

        self.train_set = LibriSpeechDataset(config, "train")
        self.val_set = LibriSpeechDataset(config, "val")
        self.train_loader = self.loader(self.train_set)
        self.val_loader = self.loader(self)
        self.processor = TextTransformer()

        self.model = create_model(model=config['model'], input_size=config['spec_params'])

    def loader(self, dataset):
        if self.world_size:
            sampler = DistributedSampler(dataset=dataset, rank=self.device, num_replicas=self.world_size)
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=custom_collate_fn)
        else:
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn,
                                num_workers=self.num_workers)

        return loader
