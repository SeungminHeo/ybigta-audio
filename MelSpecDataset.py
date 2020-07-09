from pathlib import Path
import json
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np


class MelSpecDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            target_json_path: Path,
            ):
        self.data_dir = data_path
        with open(data_path, "rb") as fp:
            self.data_files = pickle.load(fp)
        with open(target_json_path, 'r', encoding='utf-8') as fb:
            self.target_dict = json.load(fb)

    def __getitem__(self, item_index: int):
        audio_data = np.load(self.data_files[item_index])
        inputs = torch.from_numpy(audio_data)
        targets = torch.from_numpy(np.array(self.get_target(item_index))).to(dtype=torch.long)
        return inputs, targets

    def __len__(self):
        return len(self.data_files)

    def get_target(self, item_index: int):
        return self.target_dict[self.data_files[item_index].split("/")[-1].split(".")[0]]
