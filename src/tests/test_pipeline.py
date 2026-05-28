import unittest
import torch
import numpy as np
from src.core.config import Config
from src.core.model import get_model
from src.core.data_loader import haversine_distance
from src.training.train import train_model, _build_adjacency
from torch.utils.data import DataLoader, TensorDataset

class TestFraudDetection(unittest.TestCase):
    
    def test_haversine(self):
        ny = (40.7128, -74.0060)
        lon = (51.5074, -0.1278)
        dist = haversine_distance(ny[0], ny[1], lon[0], lon[1])
        self.assertGreater(dist, 5000)
        self.assertLess(dist, 6000)

    def test_model_forward(self):
        model = get_model('standard')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_vae_forward(self):
        model = get_model('vae')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        recon, mu, logvar = model(dummy_input)
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(mu.shape, (4, Config.LATENT_DIM))

    def test_contrastive_forward(self):
        model = get_model('contrastive')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        recon, z = model(dummy_input)
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(z.shape, (4, Config.LATENT_DIM))

    def test_denoising_forward(self):
        model = get_model('denoising')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        model.eval()
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_attention_forward(self):
        model = get_model('attention_ae')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_mc_dropout_forward(self):
        model = get_model('mc_dropout')
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_lstm_forward(self):
        model = get_model('lstm')
        dummy_input = torch.randn(2, Config.SEQ_LEN, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_transformer_forward(self):
        model = get_model('transformer')
        dummy_input = torch.randn(2, Config.SEQ_LEN, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_graph_forward(self):
        try:
            model = get_model('graph')
        except ImportError:
            self.skipTest("torch-geometric not installed")
            return
            
        dummy_input = torch.randn(4, Config.INPUT_DIM).to(Config.DEVICE)
        # Create a dummy edge_index for 4 nodes
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long).to(Config.DEVICE)
        recon, z = model(dummy_input, edge_index)
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(z.shape, (4, Config.LATENT_DIM))

    def test_build_adjacency(self):
        data = torch.randn(4, Config.INPUT_DIM)
        adj = _build_adjacency(data)
        self.assertEqual(adj.shape, (4, 4))
        row_sums = adj.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(4), atol=1e-5))

    def test_train_model_basic(self):
        model = get_model('standard')
        data = torch.randn(16, Config.INPUT_DIM)
        ds = TensorDataset(data, data)
        dl = DataLoader(ds, batch_size=8)
        history = train_model(model, dl, dl, epochs=2)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)

    def test_train_model_early_stopping(self):
        model = get_model('standard')
        data = torch.randn(16, Config.INPUT_DIM)
        ds = TensorDataset(data, data)
        dl = DataLoader(ds, batch_size=8)
        history = train_model(model, dl, dl, epochs=5)
        # Early stopping may trigger before 5 epochs if val_loss doesn't improve
        self.assertLessEqual(len(history['train_loss']), 5)
        self.assertGreater(len(history['train_loss']), 0)

    def test_train_model_custom_lr(self):
        model = get_model('standard')
        data = torch.randn(16, Config.INPUT_DIM)
        ds = TensorDataset(data, data)
        dl = DataLoader(ds, batch_size=8)
        history = train_model(model, dl, dl, epochs=2, lr=0.01)
        self.assertEqual(len(history['train_loss']), 2)

    def test_ensemble_scoring(self):
        from src.evaluation.evaluate import ensemble_scoring
        model1 = get_model('standard')
        model2 = get_model('vae')
        data = torch.randn(8, Config.INPUT_DIM)
        ds = TensorDataset(data, data)
        dl = DataLoader(ds, batch_size=4)
        scores = ensemble_scoring({'standard': model1, 'vae': model2}, dl)
        self.assertEqual(len(scores), 8)

if __name__ == '__main__':
    unittest.main()
