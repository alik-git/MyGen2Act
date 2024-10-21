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

    def extract_features(self, video_frames, num_frames_to_process=16):
        # Determine the total number of frames in the video
        num_frames = len(video_frames)

        # Calculate indices to sample from, ensuring the first and last frames are included
        if num_frames_to_process >= num_frames:
            sampled_indices = list(range(num_frames))  # Use all frames if video is shorter
        else:
            # Evenly sample frames, ensuring first and last frames are always included
            sampled_indices = [0]  # Start with the first frame
            step = (num_frames - 1) / (num_frames_to_process - 1)
            for i in range(1, num_frames_to_process - 1):
                sampled_indices.append(round(i * step))
            sampled_indices.append(num_frames - 1)  # Ensure the last frame is included
            sampled_indices = sorted(set(sampled_indices))

        # Extract features from the sampled frames
        features = []
        for idx in sampled_indices:
            frame = video_frames[idx]
            with torch.no_grad():
                pixel_values = self.preprocess_frame(frame)
                outputs = self.model(pixel_values)
                features.append(outputs.last_hidden_state.squeeze(0).cpu().numpy())

        # Stack all frame features into a single array
        return np.stack(features), sampled_indices