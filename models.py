import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchvision
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
            nn.BatchNorm2d(features_dim*16),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*16, features_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*8, features_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*4, features_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

class GAN(L.LightningModule):
    def __init__(self, z_dim, img_channels, features_dim, lr=2e-4, batch_size=32):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lr = lr
        self.batch_size = batch_size

        self.loss_fn = nn.BCELoss()

        self.generator = Generator(self.hparams.z_dim, img_channels=img_channels, features_dim=features_dim)
        self.discriminator = Discriminator(img_channels=img_channels, features_dim=features_dim)

        self.gen_optimizer = Adam(self.generator.parameters(), lr=self.lr)
        self.dis_optimizer = Adam(self.discriminator.parameters(), lr=self.lr)

    def forward(self, X):
        return self.generator(X)
    
    def configure_optimizers(self):
            # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.90)
            return self.gen_optimizer, self.dis_optimizer

    def training_step(self, batch):
        g_opt, d_opt = self.optimizers()
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(g_opt)
        generated_imgs = self(z)

        # log sampled images
        sample_imgs = generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        real_label = torch.ones((imgs.size(0), 1), device=self.device)
        fake_label = torch.zeros((imgs.size(0), 1), device=self.device)
        real_label = real_label.type_as(imgs)
        fake_label = fake_label.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.loss_fn(self.discriminator(generated_imgs), real_label)
        self.log("g_loss", g_loss, prog_bar=True)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        
        self.untoggle_optimizer(g_opt)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(d_opt)

        # how well can it label as real?
        real_loss = self.loss_fn(self.discriminator(imgs), real_label)

        # how well can it label as fake?
        fake_loss = self.loss_fn(self.discriminator(self(z).detach()), fake_label)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
       
        self.untoggle_optimizer(d_opt)

    
    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
