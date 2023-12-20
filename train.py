from random import random

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler, DataLoader
from dataset import LibriSpeechDataset
from utils import custom_collate_fn, TextTransformer, create_model

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

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

        self.model = create_model(model=config['model'], in_channels=config['spec_params']["n_mels"],
                                  out_channels=len(self.processor.char_map)+1)
        self.model.to(self.device)

        if self.world_size:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        self.criterion = nn.CTCLoss(blank=len(self.processor.char_map))
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'])

        if self.use_onecyclelr:
            self.scheduler = self.oneCycleLR(config)

        if from_checkpoint:
            if os.path.exists(os.path.join(self.checkpoint_dir, "model_last.pt")):
                if self.world_size:
                    map_location = {"cuda:%d" % 0: "cuda:%d" % self.device}
                    self.load_checkpoint(self.checkpoint_dir, map_location)
                else:
                    self.load_checkpoint(self.checkpoint_dir, self.device)

                with (open(os.path.join(self.checkpoint_dir, "model_last.pt"), 'r') as f):
                    last_epoch = int(f.read())
                    last_batch_idx = last_epoch * len(self.train_loader) - 1
                    self.start_epoch = last_epoch + 1
                    if self.use_onecyclelr:
                        self.scheduler = self.oneCycleLR(config, last_batch_idx)

    def loader(self, dataset):
        if self.world_size:
            sampler = DistributedSampler(dataset=dataset, rank=self.device, num_replicas=self.world_size)
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=custom_collate_fn)
        else:
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn,
                                num_workers=self.num_workers)

        return loader

    def oneCycleLR(self, hparams, last_epoch=-1):

        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=float(hparams['max_lr']),
            steps_per_epoch=len(self.train_loader),
            epochs=int(hparams['epochs']),
            div_factor=float(hparams['div_factor']),
            pct_start=float(hparams['pct_start']),
            last_epoch=last_epoch
        )

        return scheduler

    def load_checkpoint(self, path, map_location=None):

        if self.world_size:
            self.model.module.state_dict(torch.load(os.path.join(path, "model_last.pt"), map_location=map_location))
        else:
            self.model.load_state_dict(os.path.join(path, "model_last.pt"), map_location=map_location)

        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimier_last.pt"), map_location=map_location))
