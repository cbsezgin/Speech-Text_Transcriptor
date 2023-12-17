import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import re
import os
import librosa

class LibriSpeechDataset(Dataset):
    def __init__(self, config, set):
        super(LibriSpeechDataset, self).__init__()
        self.config = config
        self.param = config[set]
        self.label_encoder =

