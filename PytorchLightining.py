import torch
from pytorch_lightning import LightningModule

from torch.nn.modules.conv import Conv2d
from torch.nn.modules.rnn import LSTM

class SpeakerCLSModel(LightningModule):
    def __init__(self):
        super().__init__()
        torch.nn.
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                                transform=transforms.ToTensor()), batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)