import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import torch
import torchaudio
import librosa
from torch.nn.utils.rnn import pad_sequence

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
