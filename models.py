import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning as L

from torchmetrics import Accuracy  
from torch.optim import Adam

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_dim):
        super().__init__()

        self.img_channels = img_channels
        self.features_dim = features_dim

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.img_channels, self.features_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim, self.features_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.features_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim*2, self.features_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.features_dim*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim*4, self.features_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.features_dim*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_dim):
        super().__init__()

        self.z_dim = z_dim
        self.img_channels = img_channels
        self.features_dim = features_dim

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_dim*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*16, features_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*8, features_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*4, features_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

# class GAN(L.LightningModule):
#     def __init__(self, z_dim, img_channels, features_dim, lr=2e-4, batch_size=32):
#         super().__init__()

#         self.save_hyperparameters()
#         self.automatic_optimization = False

#         self.lr = lr
#         self.batch_size = batch_size

#         self.loss_fn = nn.BCELoss()

#         self.generator = Generator(self.hparams.z_dim, img_channels=img_channels, features_dim=features_dim)
#         self.discriminator = Discriminator(img_channels=img_channels, features_dim=features_dim)

#         self.gen_optimizer = Adam(self.parameters(), lr=self.lr)
#         self.dis_optimizer = Adam(self.parameters(), lr=self.lr)

#     def forward(self, X):
#         mel, chroma, tonetz = X
#         # print(mel.shape)
#         cnn_input = np.repeat(mel[np.newaxis, ...].cpu(), 3, axis=0).cuda()
#         cnn_input = cnn_input.permute(1, 0, 2, 3)
#         # print(cnn_input.shape)
#         transform = transforms.Compose([
#            models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2.transforms()
#         ])
#         cnn_input = transform(cnn_input)
#         # print(cnn_input.shape)
#         cnn = self.cnn_model(cnn_input)
#         cnn = cnn.view(cnn.size(0), -1)
#         chroma = chroma.to(torch.float32).cuda()
        
#         # print(X.shape)
#         # X = X.view(X.size(0), X.size(2), X.size(3))

#         chroma = chroma.permute(0, 2, 1)
#         # print(chroma.shape)


#         h0 = torch.randn(1, chroma.shape[0], 512).cuda()

#         chroma, hn = self.gru(chroma, h0)

#         chroma_output = chroma[:, -1, :]

#         tonetz = tonetz.to(torch.float32).cuda()
        
#         # print(X.shape)
#         # X = X.view(X.size(0), X.size(2), X.size(3))

#         tonetz = tonetz.permute(0, 2, 1)
#         print(tonetz.shape)


#         h0 = torch.randn(1, tonetz.shape[0], 512).cuda()

#         tonetz, hn = self.gru2(tonetz, h0)

#         tonetz_output = tonetz[:, -1, :]
#         # print(cnn.shape)
#         # print(output.shape)
#         # Concatenate cnn and output along the last dimension
#         combined = torch.cat((cnn, chroma_output, tonetz_output), dim=1)

#         output = self.fc(combined)

#         # print(output.shape)

#         return output
    
#     def configure_optimizers(self):
#             scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.90)
#             return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

#     def _step(self, batch):
#         x, y = batch
#         preds = self(x)

#         # y = y.squeeze()  # Remove singleton dimensions
#         # print(y)
#         # print(preds)

#         loss = self.loss_fn(preds, y)
#         # acc = self.acc(preds, y)
#         acc_preds = torch.argmax(preds, dim=1)
#         acc = self.acc(acc_preds, y)
#         return loss, acc
    
#     def training_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log(
#             "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.log(
#             "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
#         self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

#     def test_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
#         self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
