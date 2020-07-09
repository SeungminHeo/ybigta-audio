from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

from MelSpecDataset import MelSpecDataset
from collate import collate_equal_length


class SpeakerCLSModel(LightningModule):
    def __init__(self,
                 train_data_dir: Path,
                 valid_data_dir: Path,
                 target_json_dir: Path,
                 output_size,
                 batch_size):
        super(SpeakerCLSModel, self).__init__()
        self.output_size = output_size
        self.train_data_dir = train_data_dir
        self.val_data_dir = valid_data_dir
        self.target_json_dir = target_json_dir
        self.batch_size = batch_size
        self.metric = Accuracy()

        self.conv_layer = nn.Sequential(
            # Conv2d_0
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.1),
            # Conv2d_1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.1),
            # Conv2d_2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.1),
        )
        self.lstm_layer = nn.LSTM(128, 128, num_layers=1)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(128 * 39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=128 * 39, out_features=self.output_size, bias=True)
        )

    def forward(self, x):
        sizes_x = x.size()
        x = x.view(sizes_x[0], sizes_x[2], sizes_x[1])
        x = x.unsqueeze(1)
        out = self.conv_layer(x)
        out = out.transpose(1, 2)
        sizes = out.size()
        out = out.view(sizes[0], sizes[1], sizes[2] * sizes[3])
        out, (h, c) = self.lstm_layer(out)
        out = out.view(sizes[0], -1)

        out = self.fc_layer(out)
        return out

    def train_dataloader(self):
        data_loader = DataLoader(
            MelSpecDataset(
                data_path=self.train_data_dir,
                target_json_path=self.target_json_dir
            ),
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
            # collate_fn=collate_equal_length
        )
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(
            MelSpecDataset(
                data_path=self.val_data_dir,
                target_json_path=self.target_json_dir
            ),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
            # collate_fn=collate_equal_length
        )
        return data_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = self.metric(pred, target)
        return {'loss': loss,
                "progress_bar": {"train_acc": accuracy},
                "log": {"train_acc": accuracy, 'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = self.metric(pred, target)
        return {'val_loss': loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()

        return {'log': {'val_loss': avg_loss, "val_accuracy": avg_accuracy}}
