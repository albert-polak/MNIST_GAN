from lightning.pytorch.loggers import TensorBoardLogger
from datamodule import MNISTDataModule
from models import GAN
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
logger = TensorBoardLogger("tb_logs", name="mnist_gan_v0")
dm = MNISTDataModule(data_dir='./data/', batch_size=32)
model = GAN(z_dim=100, img_channels=1, features_dim=64)
callbacks = []
callbacks.append(ModelCheckpoint(dirpath='./models'))
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=100,
    logger=logger,
    callbacks=callbacks
)
trainer.fit(model, dm)