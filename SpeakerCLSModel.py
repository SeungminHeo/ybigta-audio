import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from MelSpecDataset import MelSpecDataset


class SpeakerCLSModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Conv2d_0
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.1),

            # Conv2d_1
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.1),

            # Conv2d_2
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.1)
        )

        self.lstm_layer = nn.LSTM(128, 64, num_layers=2, bidirectional=True)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=64, out_features=10, bias=True)
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.lstm_layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

    def forward(self, x):
        return F.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}


    def train_dataloader(self, data_dir, meta_file_path):
        data_loader = DataLoader(
            dataset=MelSpecDataset(
                data_dir=data_dir,
                meta_file_path=meta_file_path
            ),
            batch_size=16
        )
        return data_loader

    def valid_dataloader(self, data_dir, meta_file_path):
        data_loader = DataLoader(
            dataset=MelSpecDataset(
                data_dir=data_dir,
                meta_file_path=meta_file_path
            )
        )
        return data_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)