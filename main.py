import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from SpeakerCLSModel import SpeakerCLSModel
from pytorch_lightning.profiler import AdvancedProfiler

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(filepath='/datadrive/checkpoint/{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.ckpt')
    model = SpeakerCLSModel(train_data_dir="/datadrive/train.txt",
                            valid_data_dir="/datadrive/val.txt",
                            target_json_dir="/datadrive/target.json",
                            output_size=7296,
                            batch_size=128)
    trainer = Trainer(gpus=1,
                      max_epochs=10,
                      limit_val_batches=1.0,
                      check_val_every_n_epoch=1,
                      checkpoint_callback=checkpoint_callback,
                      distributed_backend="dp",
                      profiler=AdvancedProfiler(),)
    trainer.fit(model)

