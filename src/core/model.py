import torch
import torch.nn as nn
from src.core.config import Config

class Autoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5
        
    def forward(self, x):
        # x: [batch, dim]
        # Simple self-attention for tabular data (feature attention)
        # Usually attention is over sequence [batch, seq, dim].
        # For tabular [batch, dim], we can treat dim as sequence length 1? No.
        # "Make your AE 'focus' on important features". 
        # We can treat features as a sequence if we unsqueeze, or learn weights per feature.
        # But standard self-attention is calculating relationship between samples in a batch? No, intra-sample.
        # Tabular self-attention: usually requires projecting features to embeddings [batch, num_feats, emb_dim].
        # Here input is just [batch, input_dim].
        # Let's implement a simple "Feature Attention" mechanism:
        # Weights features based on their importance dynamically.
        
        # Validation: We really need feature embeddings to do proper feature-wise attention.
        # As a simplification for this project:
        # Gate mechanism (Attention-like): Input * Sigmoid(MLP(Input))
        
        attention_weights = torch.sigmoid(self.query(x))
        return x * attention_weights

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(AttentionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.ReLU(),
            AttentionBlock(Config.HIDDEN_DIM1),
            nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM1),
            nn.ReLU(),
            AttentionBlock(Config.HIDDEN_DIM1),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Tanh() # Assuming scaler scales to [-1, 1] or similar? StandardScaler is mean 0 std 1. Tanh might limit range. Linear is safer for unbounded.
            # But usually we match range. Let's use Linear and rely on loss.
        )
        # Replacing last Tanh with Linear as features are standard scaled (unbounded)
        self.model[-1] = nn.Linear(128, output_dim)

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(ContrastiveAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM),
        )
        # Projection head for contrastive loss
        self.projector = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        recon = self.decoder(h)
        return recon, z

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Very simple GCN: D^-1/2 A D^-1/2 X W
        # Simplified for tabular where adj might be dynamic: A X W
        support = self.linear(x)
        output = torch.mm(adj, support)
        return output

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(GraphAutoencoder, self).__init__()
        self.gc1 = GraphConvolution(input_dim, Config.HIDDEN_DIM1)
        self.gc2 = GraphConvolution(Config.HIDDEN_DIM1, Config.LATENT_DIM)
        self.dc1 = GraphConvolution(Config.LATENT_DIM, Config.HIDDEN_DIM1)
        self.dc2 = GraphConvolution(Config.HIDDEN_DIM1, input_dim)

    def forward(self, x, adj):
        h1 = torch.relu(self.gc1(x, adj))
        z = torch.relu(self.gc2(h1, adj))
        h2 = torch.relu(self.dc1(z, adj))
        recon = self.dc2(h2, adj)
        return recon, z

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, Config.HIDDEN_DIM1)
        self.fc2_mu = nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM)
        self.fc2_logvar = nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM)
        
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class DenoisingAutoencoder(Autoencoder):
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * Config.NOISE_FACTOR
            x = x + noise
        return super().forward(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, hidden_dim=32):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: [batch, seq, input_dim] -> [batch, hidden_dim]
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, Config.LATENT_DIM)
        
        # Decoder: [batch, latent_dim] -> [batch, seq, input_dim]
        self.from_latent = nn.Linear(Config.LATENT_DIM, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        latent = torch.relu(self.to_latent(h_n.squeeze(0)))
        
        # Decode
        h_0_dec = self.from_latent(latent).unsqueeze(0)
        c_0_dec = torch.zeros_like(h_0_dec)
        
        # Repeat latent or use zeros as input
        # Standard: use zeros or previous output. Here zeros for simplicity.
        dec_input = torch.zeros(batch_size, seq_len, self.hidden_dim).to(Config.DEVICE)
        
        dec_out, _ = self.decoder_lstm(dec_input, (h_0_dec, c_0_dec))
        
        # [batch, seq, hidden_dim] -> [batch, seq, input_dim]
        recon = self.output_layer(dec_out)
        return recon

class MCDropoutAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, dropout_rate=0.2):
        super(MCDropoutAutoencoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(Config.HIDDEN_DIM1, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_model(model_type='standard'):
    if model_type == 'standard':
        return Autoencoder().to(Config.DEVICE)
    elif model_type == 'vae':
        return VariationalAutoencoder().to(Config.DEVICE)
    elif model_type == 'denoising':
        return DenoisingAutoencoder().to(Config.DEVICE)
    elif model_type == 'lstm':
        return LSTMAutoencoder().to(Config.DEVICE)
    elif model_type == 'mc_dropout':
        return MCDropoutAutoencoder().to(Config.DEVICE)
    elif model_type == 'attention_ae':
        return AttentionAutoencoder().to(Config.DEVICE)
    elif model_type == 'contrastive':
        return ContrastiveAutoencoder().to(Config.DEVICE)
    elif model_type == 'graph':
        return GraphAutoencoder().to(Config.DEVICE)
    else:
        raise ValueError("Invalid model type")
