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
import torch.nn.functional as F

from matplotlib import pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_dim):
        super().__init__()

        self.img_channels = img_channels
        self.features_dim = features_dim

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.img_channels, self.features_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim, self.features_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.features_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim*2, self.features_dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.features_dim*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.features_dim*4, self.features_dim*8, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.ConvTranspose2d(z_dim, features_dim*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_dim*16),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*16, features_dim*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*8, features_dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_dim*4, features_dim, kernel_size=4, stride=2, padding=1, bias=False),
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

        self.generator = Generator(self.hparams.z_dim, img_channels=img_channels, features_dim=features_dim)
        self.discriminator = Discriminator(img_channels=img_channels, features_dim=features_dim)

        self.gen_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.dis_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.initialize_weights(self.generator)
        self.initialize_weights(self.discriminator)

        self.validation_z = torch.randn(32, self.hparams.z_dim, 1, 1)

        self.step = 0
        
    def initialize_weights(self, model):
        # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, X):
        return self.generator(X)
    
    def configure_optimizers(self):
            # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.90)
            return [self.gen_optimizer, self.dis_optimizer]

    def adversarial_loss(self, y_hat, y):
        if y.any() == 1:
            y_smoothed = torch.rand_like(y_hat) * 0.1 + 0.9  # Smoothing for real samples
        else:
            y_smoothed = torch.rand_like(y_hat) * 0.1  # Smoothing for fake samples

        # Compute binary cross-entropy loss with smoothed labels
        return F.binary_cross_entropy(y_hat, y_smoothed)

    def training_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        g_opt, d_opt = self.optimizers()
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.z_dim, 1, 1)
        # z.requires_grad_(True)
        z = z.type_as(imgs)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = self(z.type_as(self.generator.generator[0].weight))
                grid = torchvision.utils.make_grid(fake)
                self.logger.experiment.add_image("generated_fakes_step", grid, self.step)

        if batch_idx % 2 == 0:
            self.toggle_optimizer(g_opt)
            fake = self(z)
            
            # adversarial loss is binary cross-entropy
            output = self.discriminator(fake).reshape(-1)
            g_loss = self.adversarial_loss(output, torch.ones_like(output))
            self.log("g_loss", g_loss, prog_bar=True)
            self.generator.zero_grad()
            
            self.manual_backward(g_loss)
            g_opt.step()
            self.untoggle_optimizer(g_opt)
            
        else:
            self.toggle_optimizer(d_opt)

            fake = self(z)
            disc_real = self.discriminator(imgs).reshape(-1)
            disc_fake = self.discriminator(fake.detach()).reshape(-1)

            # how well can it label as real?
            real_loss = self.adversarial_loss(disc_real, torch.ones_like(disc_real))

            # how well can it label as fake?
            fake_loss = self.adversarial_loss(disc_fake, torch.zeros_like(disc_fake))

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            self.discriminator.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
            self.untoggle_optimizer(d_opt)
        self.step+= 1


    def on_train_epoch_end(self):
        with torch.no_grad():
            z = self.validation_z.type_as(self.generator.generator[0].weight)
            # log sampled images
            sample_imgs = self(z)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
