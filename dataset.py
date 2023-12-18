import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import re
import os
import librosa
from utils import TextTransformer, audio_to_mel

class LibriSpeechDataset(Dataset):
    def __init__(self, config, set):
        super(LibriSpeechDataset, self).__init__()
        self.config = config
        self.param = config[set]
        self.label_encoder = TextTransformer()

        if not os.path.exists(self.param['data_list']):
            self.create_data_list()

        if self.config["normalize"]:
            if os.path.exists(self.config["stats"]):
                stats = torch.from_numpy(np.load(self.config["stats"])).permute(1,0)
                if stats.shape[0] == 1:
                    self.mean = stats[0,0]
                    self.std = stats[0,1]
                else:
                    self.mean = stats[:,0].unsqueeze()
                    self.std = stats[:,1].unsqueeze()

        with open(self.param["data_list"], "r") as f:
            data = f.readlines()
        data = [line.strip().split() for line in data]
        self.collection = data

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        audio, transcription = self.collection[item]

        audio, _ = librosa.load(audio, sr=self.config["spec_params"]["sr"])
        with open(transcription, "r") as f:
            transcription = f.read()
            transcription = transcription.lower()
            transcription = re.sub("[^a-zA-Z0-9]+", "", transcription)
            transcription = torch.tensor(self.label_encoder.text_to_int_sequence(transcription), dtype=torch.long)

        if self.param.get("apply_speed_perturbation", None):
            limit = self.config.get("speed_perturbation", 0.1)
            rate = np.random.uniform(low=1-limit, high=1+limit)
            audio = librosa.effects.time_stretch(audio, rate)

        mel_spectrogram = audio_to_mel(audio, self.config["spec_params"])

        if self.config["normalize"]:
            mel_spectrogram = normalize(mel_spectrogram)

        if self.param.get("apply_masking", None):
            # mel_spectrogram =



    def create_data_list(self):
        data_list = open(self.param['data_dir'], "w")

        for folder in self.param['data_dir']:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith((".wav", ".flac")):
                        audio, sr = librosa.load(os.path.join(root, file), sr=self.config["spec_params"]["sr"])
                        if self.config.get("max_length", None):
                            length = audio.shape[0] / sr
                            if length > int(self.config["max_length"]):
                                continue
                        label = os.path.splitext(file)[0] + "normalized.txt"
                        if not os.path.exists(os.path.join(root, label)):
                            continue
                        data_list.write(f"{os.path.join(root, file)} {os.path.join(root, label)}\n")

        data_list.close()


