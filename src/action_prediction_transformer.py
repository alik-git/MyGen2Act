import torch
import torch.nn as nn
import numpy as np

class ActionPredictionTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_encoder_layers=6, num_decoder_layers=6, num_heads=8, 
                 action_dim=7, num_future_actions=4, num_bins=256, dropout=0.1):
        super(ActionPredictionTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_future_actions = num_future_actions
        self.action_dim = action_dim
        self.num_bins = num_bins

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(hidden_dim, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection to predict action bins
        self.output_proj = nn.Linear(hidden_dim, num_future_actions * action_dim * num_bins)

    def forward(self, z_g, z_r):
        # z_g: [batch_size, seq_len_enc, hidden_dim]
        # z_r: [batch_size, seq_len_dec, hidden_dim]

        # Add positional encoding
        z_g = self.positional_encoder(z_g.permute(1, 0, 2))  # Shape: [seq_len_enc, batch_size, hidden_dim]
        z_r = self.positional_encoder(z_r.permute(1, 0, 2))  # Shape: [seq_len_dec, batch_size, hidden_dim]

        # Encoder
        memory = self.encoder(z_g)  # Shape: [seq_len_enc, batch_size, hidden_dim]

        # Decoder
        decoder_output = self.decoder(z_r, memory)  # Shape: [seq_len_dec, batch_size, hidden_dim]

        # Take the mean over sequence length
        decoder_output = decoder_output.mean(dim=0)  # Shape: [batch_size, hidden_dim]

        # Project to action logits
        action_logits = self.output_proj(decoder_output)  # Shape: [batch_size, num_future_actions * action_dim * num_bins]

        # Reshape to [batch_size, num_future_actions, action_dim, num_bins]
        action_logits = action_logits.view(
            -1, self.num_future_actions, self.action_dim, self.num_bins
        )

        return action_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model).to(torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)