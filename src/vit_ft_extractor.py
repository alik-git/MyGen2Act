import torch
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel


# Define the ViT encoder (using ViT-B/16)
class VideoFeatureExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda:0'):
        self.device = device
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_frame(self, frame):
        # Ensure frame values are in [0, 1] before feeding to the feature extractor
        frame = (frame - frame.min()) / (frame.max() - frame.min())

        # Apply the same preprocessing as the ViT feature extractor
        inputs = self.feature_extractor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        return pixel_values

    def extract_features(self, video_frames, sampled_indices=None):
        """
        Extracts features from the specified sampled indices of the video frames.

        Args:
        - video_frames: Torch tensor of shape (batch_size, num_frames, 3, H, W).
        - sampled_indices: List of indices indicating which frames to process.

        Returns:
        - features: Torch tensor of shape (batch_size, num_frames, num_tokens, hidden_dim).
        """
        batch_size, num_frames, _, _, _ = video_frames.shape

        # Ensure sampled indices are within the range of available frames
        if sampled_indices is None:
            sampled_indices = list(range(num_frames))
        else:
            sampled_indices = [idx for idx in sampled_indices if 0 <= idx < num_frames]

        features = []
        for idx in sampled_indices:
            frame_batch = video_frames[:, idx, :, :, :]  # shape: (batch_size, 3, H, W)
            with torch.no_grad():
                pixel_values = self.preprocess_frame(frame_batch)
                outputs = self.model(pixel_values)
                features.append(outputs.last_hidden_state)  # shape: (batch_size, num_tokens, hidden_dim)

        # Stack features along the time dimension
        features = torch.stack(features, dim=1)  # shape: (batch_size, num_frames, num_tokens, hidden_dim)

        return features