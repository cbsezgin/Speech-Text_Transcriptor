import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import torch
import torchaudio
import librosa
from torch.nn.utils.rnn import pad_sequence

from models import QuartzNet


class TextTransformer():
    def __init__(self):
        self.char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}

        for line in self.char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int[index]] = ch

        self.index_map[1] = " "

    def text_to_int_sequence(self, text):
        int_sequence = []

        for char in text:
            if char == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[char]
            int_sequence.append(ch)

        return int_sequence

    def int_to_text(self, int_sequence):
        string = []

        for int_char in int_sequence:
            string.append(self.index_map[int_char])
        return "".join(string)

    def decode_sequence(self, output, labels, label_len, blank_label=28, collapse_pad=True):
        argmax = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, arg in enumerate(argmax):
            dec = []
            targets.append(self.int_to_text(labels[i][:label_len[i]].to_list()))

            for j, index in enumerate(arg):
                if index != blank_label:
                    if collapse_pad and (j != 0) and (index == arg[j-1]):
                        continue
                    dec.append(index.item())
            decodes.append(self.int_to_text(dec))
        return decodes, targets

def audio_to_mel(x, hparams):
    spec = librosa.feature.melspectrogram(
        x,
        sr=hparams["sr"],
        n_fft=hparams["n_fft"],
        win_length=hparams["win_length"],
        hop_length=hparams["hop_length"],
        n_mels=hparams["n_mels"],
        power=1,
        fmin=0,
        fmax=8000
    )

    spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))
    spec = torch.FloatTensor(spec)

    return spec


def augment(spec, chunk_size=30, freq_mask_param=10, time_mask_param=6):
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(freq_mask_param), iid_masks=True)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=int(time_mask_param), iid_masks=True)
    num_chunks = spec.shape[1] // int(chunk_size)

    if num_chunks <= 1:
        return spec
    else:
        chunks = torch.split(spec, num_chunks, dim=1)
        to_be_masked = torch.stack(list(chunks[:-1]), dim=1).unsqueeze(1)
        time_mask(to_be_masked)
        freq_mask(to_be_masked)
        masked = to_be_masked.squeeze(1).permute(1,0,2).reshape(spec.shape[0], -1)
        return torch.cat([masked, chunks[-1]], dim=1)


def custom_collate_fn(data):
    melpecs, texts, input_lengths, target_length = zip(*data)

    specs = [torch.transpose(spec, 0, 1) for spec in melpecs]
    specs = pad_sequence(specs, batch_first=True)
    specs = torch.transpose(specs, 1, 2)
    labels = pad_sequence(texts, batch_first=True)

    return specs, labels, torch.tensor(input_lengths), torch.tensor(target_length)


def create_model(model, in_channels, out_channels):
    models = ["quartznet5x5", "quartznet10x5", "quartznet15x5"]
    assert (model in models), f"Unknown model: {model}"

    if model == "quartznet5x5":
        return QuartzNet(repeat=0, in_channels=in_channels, out_channels=out_channels)

    elif model == "quartznet10x5":
        return QuartzNet(repeat=1, in_channels=in_channels, out_channels=out_channels)

    elif model == "quartznet15x5":
        return QuartzNet(repeat=2, in_channels=in_channels, out_channels=out_channels)
