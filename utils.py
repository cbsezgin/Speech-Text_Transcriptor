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