import unittest
import torch
import numpy as np
import pandas as pd
from src.core.config import Config
from src.core.model import get_model
from src.core.data_loader import haversine_distance

class TestFraudDetection(unittest.TestCase):
    
    def test_haversine(self):
        # New York to London approx
        ny = (40.7128, -74.0060)
        lon = (51.5074, -0.1278)
        dist = haversine_distance(ny[0], ny[1], lon[0], lon[1])
        self.assertGreater(dist, 5000)
        self.assertLess(dist, 6000)

    def test_model_forward(self):
        model = get_model('standard')
        dummy_input = torch.randn(1, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_vae_forward(self):
        model = get_model('vae')
        dummy_input = torch.randn(1, Config.INPUT_DIM).to(Config.DEVICE)
        recon, mu, logvar = model(dummy_input)
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(mu.shape, (1, Config.LATENT_DIM))

    def test_contrastive_forward(self):
        model = get_model('contrastive')
        dummy_input = torch.randn(1, Config.INPUT_DIM).to(Config.DEVICE)
        recon, z = model(dummy_input)
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(z.shape, (1, Config.LATENT_DIM))

if __name__ == '__main__':
    unittest.main()
