from MelSpecDataset import MelSpecDataset

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':

    data_loader = DataLoader(
        dataset=MelSpecDataset(
            data_dir="/home/seungmin/ybigta-conf/data",
            meta_file_path="/home/seungmin/yㅌㅈbigta-conf"
        ),
        batch_size=16
    )

    for data in data_loader:
        print(data[0].shape)
        print(data[1].shape)
