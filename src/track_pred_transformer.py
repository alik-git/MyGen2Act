import torch
import torch.nn as nn

class TrackPredictionTransformer(nn.Module):
    def __init__(self, point_dim=2, hidden_dim=768, num_layers=6, num_heads=8, num_frames=16):
        super(TrackPredictionTransformer, self).__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.point_embedding = nn.Linear(point_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, num_frames * 2)  # Predict x,y for each frame

    def forward(self, P0, i0_g, zg):
        # P0: [batch_size, num_points, 2]
        # i0_g: [batch_size, num_tokens, hidden_dim]
        # zg: [batch_size, num_latents, hidden_dim]
        batch_size, num_points, _ = P0.shape

        # # Print input shapes
        # print(f"P0 shape: {P0.shape}")
        # print(f"i0_g shape: {i0_g.shape}")
        # print(f"zg shape: {zg.shape}")

        # # Print model parameters' sizes
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"Total model parameters: {total_params}")
        # for name, param in self.named_parameters():
        #     print(f"{name}: {param.shape} - {param.numel() / 1e6:.2f}M parameters")

        # Embed P0
        point_embeds = self.point_embedding(P0)  # [batch_size, num_points, hidden_dim]
        # Ensure i0_g and zg have the same batch size
        if i0_g.shape[0] != batch_size:
            i0_g = i0_g.expand(batch_size, -1, -1)
        if zg.shape[0] != batch_size:
            zg = zg.expand(batch_size, -1, -1)
        # Concatenate point embeddings, i0_g, and zg along sequence dimension
        inputs = torch.cat([point_embeds, i0_g, zg], dim=1)  # [batch_size, seq_len, hidden_dim]
        # Permute to [sequence_length, batch_size, hidden_dim] for transformer
        inputs = inputs.permute(1, 0, 2)
        outputs = self.transformer(inputs)  # [sequence_length, batch_size, hidden_dim]
        # Take outputs corresponding to the point embeddings
        point_outputs = outputs[:num_points]  # [num_points, batch_size, hidden_dim]
        # Permute back to [batch_size, num_points, hidden_dim]
        point_outputs = point_outputs.permute(1, 0, 2)
        # Predict tracks
        track_predictions = self.output_layer(point_outputs)  # [batch_size, num_points, num_frames * 2]
        # Reshape to [batch_size, num_points, num_frames, 2]
        track_predictions = track_predictions.view(batch_size, num_points, self.num_frames, 2)
        return track_predictions

if __name__ == "__main__":
    device = 'cuda:0'

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define model hyperparameters
    batch_size = 1
    num_points = 1024
    num_tokens = 197
    num_latents = 64
    hidden_dim = 768
    num_frames = 16

    # Initialize random input tensors with the expected shapes
    P0 = torch.randn(batch_size, num_points, 2, device=device)
    i0_g = torch.randn(batch_size, num_tokens, hidden_dim, device=device)
    zg = torch.randn(batch_size, num_latents, hidden_dim, device=device)

    # Initialize the model
    model = TrackPredictionTransformer(
        point_dim=2, hidden_dim=hidden_dim, num_layers=6, num_heads=8, num_frames=num_frames
    ).to(device)

    # Run a forward pass
    with torch.no_grad():
        track_predictions = model(P0, i0_g, zg)

    # Print final output shape
    print(f"Final track predictions shape: {track_predictions.shape}")
