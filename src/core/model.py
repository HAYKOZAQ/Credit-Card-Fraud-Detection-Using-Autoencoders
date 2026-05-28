import torch
import torch.nn as nn
from src.core.config import Config

class Autoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, self.num_heads, self.head_dim) for t in qkv]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.view(batch_size, -1)
        return self.out_proj(context) + x

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(AttentionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            AttentionBlock(Config.HIDDEN_DIM1),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            AttentionBlock(Config.HIDDEN_DIM2),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
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
            nn.Linear(128, output_dim)
        )

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
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM),
        )
        # Projection head for contrastive loss
        self.projector = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        recon = self.decoder(h)
        return recon, z

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.mm(adj, support)
        return output

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(GraphAutoencoder, self).__init__()
        if GCNConv is None:
            raise ImportError("Please install torch-geometric to use GraphAutoencoder.")
            
        self.gc1 = GCNConv(input_dim, Config.HIDDEN_DIM1)
        self.gc2 = GCNConv(Config.HIDDEN_DIM1, Config.LATENT_DIM)
        self.dc1 = GCNConv(Config.LATENT_DIM, Config.HIDDEN_DIM1)
        self.dc2 = GCNConv(Config.HIDDEN_DIM1, input_dim)

    def forward(self, x, edge_index):
        h1 = torch.relu(self.gc1(x, edge_index))
        z = torch.relu(self.gc2(h1, edge_index))
        h2 = torch.relu(self.dc1(z, edge_index))
        recon = self.dc2(h2, edge_index)
        return recon, z

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, Config.HIDDEN_DIM1)
        self.bn1 = nn.BatchNorm1d(Config.HIDDEN_DIM1)
        self.fc2 = nn.Linear(Config.HIDDEN_DIM1, Config.HIDDEN_DIM2)
        self.bn2 = nn.BatchNorm1d(Config.HIDDEN_DIM2)
        self.fc2_mu = nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM)
        self.fc2_logvar = nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM)
        
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM2, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM1, input_dim)
        )

    def encode(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1)))
        return self.fc2_mu(h2), self.fc2_logvar(h2)

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
        
        _, (h_n, c_n) = self.encoder_lstm(x)
        latent = torch.relu(self.to_latent(h_n.squeeze(0)))
        
        h_0_dec = self.from_latent(latent).unsqueeze(0)
        c_0_dec = c_n
        
        hidden_input = h_0_dec.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder_lstm(hidden_input, (h_0_dec, c_0_dec))
        
        recon = self.output_layer(dec_out)
        return recon

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, d_model=32, nhead=4, num_layers=2):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.to_latent = nn.Linear(d_model, Config.LATENT_DIM)
        
        # Decoder
        self.from_latent = nn.Linear(Config.LATENT_DIM, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Project back to original input dimensions
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # Project input: [batch, seq_len, d_model]
        x_proj = self.input_projection(x)
        
        # Encode: [batch, seq_len, d_model]
        memory = self.transformer_encoder(x_proj)
        
        # Pool to get latent representation (using mean pooling over sequence)
        pooled_memory = memory.mean(dim=1)
        latent = torch.relu(self.to_latent(pooled_memory))
        
        # Decode
        latent_expanded = self.from_latent(latent).unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out = self.transformer_decoder(latent_expanded, memory)
        
        # Project back to input dim
        recon = self.output_projection(dec_out)
        
        return recon


class MCDropoutAutoencoder(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, dropout_rate=0.3):
        super(MCDropoutAutoencoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(Config.HIDDEN_DIM1, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(Config.HIDDEN_DIM2, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM2),
            nn.BatchNorm1d(Config.HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(Config.HIDDEN_DIM2, Config.HIDDEN_DIM1),
            nn.BatchNorm1d(Config.HIDDEN_DIM1),
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
    elif model_type == 'transformer':
        return TransformerAutoencoder().to(Config.DEVICE)
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