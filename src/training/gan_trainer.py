import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.core.config import Config
from src.core.model import Generator, Discriminator

class GANTrainer:
    def __init__(self, fraud_loader):
        self.loader = fraud_loader
        self.latent_dim = Config.LATENT_DIM * 2 # Little bigger for generator
        self.input_dim = Config.INPUT_DIM
        
        self.generator = Generator(self.latent_dim, self.input_dim).to(Config.DEVICE)
        self.discriminator = Discriminator(self.input_dim).to(Config.DEVICE)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        
    def train(self, epochs=50):
        print("Starting GAN Training (minority oversampling)...")
        self.generator.train()
        self.discriminator.train()
        d_loss = g_loss = None
        
        for epoch in range(epochs):
            for real_samples, _ in self.loader:
                batch_size = real_samples.size(0)
                real_samples = real_samples.to(Config.DEVICE)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(Config.DEVICE)
                fake_labels = torch.zeros(batch_size, 1).to(Config.DEVICE)
                
                # --- Train Discriminator ---
                self.d_optimizer.zero_grad()
                
                outputs_real = self.discriminator(real_samples)
                d_loss_real = self.criterion(outputs_real, real_labels)
                
                z = torch.randn(batch_size, self.latent_dim).to(Config.DEVICE)
                fake_samples = self.generator(z)
                outputs_fake = self.discriminator(fake_samples.detach())
                d_loss_fake = self.criterion(outputs_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # --- Train Generator ---
                self.g_optimizer.zero_grad()
                
                outputs_fake_for_g = self.discriminator(fake_samples)
                g_loss = self.criterion(outputs_fake_for_g, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
            if (epoch+1) % 10 == 0 and d_loss is not None:
                print(f"GAN Epoch {epoch+1}: D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")
                
    def generate_synthetics(self, num_samples):
        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim).to(Config.DEVICE)
        with torch.no_grad():
            syn = self.generator(z)
        return syn.cpu().numpy()
