from lightning.pytorch.loggers import TensorBoardLogger
from datamodule import MNISTDataModule
from models import GAN
import lightning as L

logger = TensorBoardLogger("tb_logs", name="mnist_gan_v0")
dm = MNISTDataModule()
model = GAN(z_dim=100, img_channels=1, features_dim=64)
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=100,
    logger=logger,
)
trainer.fit(model, dm)