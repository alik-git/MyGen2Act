import os
import torch
from torch import nn
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel


# Define the ViT encoder (using ViT-B/16)
class VideoFeatureExtractor(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda:0', resize_height=224, resize_width=224):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.resize_height = resize_height
        self.resize_width = resize_width

        # Initialize the ViT model and feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.to(self.device)

    def preprocess(self, frames):
        """
        Preprocess frames: resize, normalize, and move to device.

        Args:
        - frames (numpy array): Input frames of shape (batch_size, num_frames, H, W, C).

        Returns:
        - torch.Tensor: Preprocessed frames of shape (batch_size, num_frames, 3, resize_height, resize_width).
        """
        batch_size, num_frames, H, W, C = frames.shape
        
        # Shape assertion
        assert frames.ndim == 5, f"Expected 5D input, got {frames.ndim}D."
        assert C == 3, f"Expected 3 channels, got {C}."
        
        frames = frames.reshape(-1, H, W, C)  # Flatten batch and time dimensions

        # Convert to torch tensor and normalize to [0, 1]
        frames = torch.from_numpy(frames).float().div(255).to(self.device)
        frames = frames.permute(0, 3, 1, 2)  # (batch_size * num_frames, C, H, W)

        # Resize frames only if necessary
        if (H, W) != (self.resize_height, self.resize_width):
            frames = torch.nn.functional.interpolate(
                frames, size=(self.resize_height, self.resize_width), mode='bilinear', align_corners=False
            )

        # Normalize frames using mean and std of 0.5 (specific to this ViT model)
        # See the documentation here https://huggingface.co/google/vit-base-patch16-224
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        frames = frames.to(self.device)  # Move frames to the device before normalization
        frames = (frames - mean) / std

        # Reshape back to (batch_size, num_frames, 3, resize_height, resize_width)
        return frames.view(batch_size, num_frames, 3, self.resize_height, self.resize_width)

    def forward(self, video_frames):
        """
        Extract features from all frames of the video batch.

        Args:
        - video_frames: Numpy array of shape (batch_size, num_frames, H, W, C).

        Returns:
        - features: Torch tensor of shape (batch_size, num_frames, num_tokens, hidden_dim).
        """
        # Preprocess the frames
        video_frames = self.preprocess(video_frames)

        # Flatten batch and frame dimensions to process all frames at once
        batch_size, num_frames, _, H, W = video_frames.shape
        
        # Shape assertions
        assert video_frames.ndim == 5, f"Expected 5D input after preprocessing, got {video_frames.ndim}D."
        assert H == 224 and W == 224, f"Expected frames of size 224x224, got {H}x{W}."

        video_frames = video_frames.view(-1, 3, H, W)  # (batch_size * num_frames, 3, H, W)

        # Forward pass through the ViT model
        with torch.no_grad():
            inputs = {'pixel_values': video_frames}
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state  # (batch_size * num_frames, num_tokens, hidden_dim)

        # Reshape back to (batch_size, num_frames, num_tokens, hidden_dim)
        features = features.view(batch_size, num_frames, features.shape[1], features.shape[2])

        return features
        
    def save(self, save_dir):
        """
        Save the ViT model, ViT feature extractor, and model name separately.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save ViT model state dict
        vit_model_path = os.path.join(save_dir, 'vit_model.pth')
        torch.save(self.model.state_dict(), vit_model_path)

        # Save ViT feature extractor configuration
        vit_feat_extractor_dir = os.path.join(save_dir, 'vit_feature_extractor')
        self.feature_extractor.save_pretrained(vit_feat_extractor_dir)

        # Save model name to a text file
        model_name_path = os.path.join(save_dir, 'model_name.txt')
        with open(model_name_path, 'w') as f:
            f.write(self.model_name)

        print(f"Saved ViT model to {vit_model_path}")
        print(f"Saved ViT feature extractor to {vit_feat_extractor_dir}")
        print(f"Saved model name to {model_name_path}")

    @classmethod
    def from_pretrained(cls, load_dir, device='cuda:0'):
        """
        Load the ViT model, ViT feature extractor, and model name separately.
        """
        # Load model name from the text file
        model_name_path = os.path.join(load_dir, 'model_name.txt')
        with open(model_name_path, 'r') as f:
            model_name = f.read().strip()

        # Create an instance of VideoFeatureExtractor
        model = cls(model_name=model_name, device=device)

        # Load ViT model state dict
        vit_model_path = os.path.join(load_dir, 'vit_model.pth')
        model.model.load_state_dict(torch.load(vit_model_path, map_location=device))

        # Load ViT feature extractor configuration
        vit_feat_extractor_dir = os.path.join(load_dir, 'vit_feature_extractor')
        model.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_feat_extractor_dir)

        model.to(device)
        print(f"Loaded ViT model from {vit_model_path}")
        print(f"Loaded ViT feature extractor from {vit_feat_extractor_dir}")
        print(f"Loaded model name from {model_name_path}")

        return model