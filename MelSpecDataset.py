from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import numpy as np


class MelSpecDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
            meta_file_path: Path,
            ):
        self.data_dir = data_dir
        with open(f'{meta_file_path}/song_meta.json', 'r', encoding='utf-8') as fb:
            self.song_meta = json.load(fb)

    def __getitem__(self, item_index: int):
        inputs = torch.FloatTensor(np.load(f"{self.data_dir}/{self.get_song_id(item_index)}.npy"))
        targets = torch.FloatTensor([self.get_artist_id(item_index)])
        return inputs, targets

    def __len__(self):
        return len(self.song_meta)

    def get_artist_id(self, item_index: int):
        return self.song_meta[item_index]["artist_id"]

    def get_song_id(self, item_index: int):
        return int(self.song_meta[item_index]["id"])
