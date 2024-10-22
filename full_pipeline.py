import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import mediapy as media
import numpy as np
import tensorflow_datasets as tfds

import os

import argparse
from pathlib import Path
from PIL import Image

from src.vit_ft_extractor import VideoFeatureExtractor
from src.perceiver_resampler import PerceiverResampler
from src.track_pred_transformer import TrackPredictionTransformer

from src.utils import construct_episode_label
from src.utils import get_sample_indices

# Helper functions
def preprocess_frames(frames, resize_height=224, resize_width=224):
    """
    Preprocess a list of frames: resize and normalize.
    """
    transform = T.Compose([
        T.Resize((resize_height, resize_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_frames = [transform(frame).numpy() for frame in frames]
    return processed_frames

def load_tracking_results(tracks_path, visibles_path):
    """
    Load tracks and visibles from .npz files.
    """
    tracks = np.load(tracks_path)['tracks']
    visibles = np.load(visibles_path)['visibles']
    print(f"Tracks loaded from: {tracks_path}")
    print(f"Visibles loaded from: {visibles_path}")
    return tracks, visibles

def process_video(frames, tracks, sampled_indices, device, video_type='generated'):
    """
    Process a video through ViT, Perceiver Resampler, and Track Prediction Transformer.
    Returns the latent tokens, initial ViT features, initial points, and predicted tracks.
    """
    # Preprocess frames
    processed_frames = preprocess_frames(frames)
    print(f"Processed {video_type} video frames: {len(processed_frames)} frames")

    # Initialize the ViT feature extractor
    feature_extractor = VideoFeatureExtractor(device=device)

    # Extract features from video frames
    features = feature_extractor.extract_features(
        processed_frames, sampled_indices=sampled_indices
    )
    features = features[0] # Shape: [1, num_frames, num_tokens, hidden_dim]
    features = torch.tensor(features, device=device, dtype=torch.float32)
    print(f"{video_type.capitalize()} video features shape: {features.shape}")

    # Extract initial ViT features (from the first frame)
    i0 = features[0].unsqueeze(0)  # Shape: [1, num_tokens, hidden_dim]

    # Flatten features for PerceiverResampler
    num_frames_processed, num_tokens, hidden_dim = features.shape
    seq_len = num_frames_processed * num_tokens
    features_flat = features.view(1, seq_len, hidden_dim)  # Shape: [1, seq_len, hidden_dim]

    # Initialize PerceiverResampler
    perceiver_resampler = PerceiverResampler(
        dim=hidden_dim, num_latents=64, num_layers=2, num_heads=8, dim_head=64, ff_mult=4
    ).to(device)

    # Pass features through PerceiverResampler
    z = perceiver_resampler(features_flat)  # Shape: [1, num_latents, hidden_dim]
    print(f"{video_type.capitalize()} video latent tokens shape: {z.shape}")

    # Prepare initial points P0
    num_points = tracks.shape[0]  # Number of points
    P0 = tracks[:, sampled_indices[0], :]  # Shape: [num_points, 2]
    # Normalize coordinates to be between 0 and 1
    img_height, img_width = frames[0].size[1], frames[0].size[0]
    P0_normalized = P0 / np.array([img_width, img_height])
    P0_normalized = torch.tensor(P0_normalized, device=device).float().unsqueeze(0)  # [1, num_points, 2]

    # Prepare ground truth tracks τ
    gt_tracks = tracks[:, sampled_indices, :]  # [num_points, num_frames_processed, 2]
    gt_tracks_normalized = gt_tracks / np.array([img_width, img_height])
    gt_tracks_normalized = torch.tensor(gt_tracks_normalized, device=device).float().unsqueeze(0)

    # Initialize TrackPredictionTransformer
    track_predictor = TrackPredictionTransformer(
        point_dim=2, hidden_dim=hidden_dim, num_layers=6, num_heads=8, num_frames=num_frames_processed
    ).to(device)

    # Predict tracks τ̂
    predicted_tracks = track_predictor(P0_normalized, i0, z)  # [1, num_points, num_frames_processed, 2]

    return z, i0, P0_normalized, predicted_tracks, gt_tracks_normalized

def main(bridge_data_path, device='cuda:0'):
    # Load the dataset
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    dataset = dataset_builder.as_dataset(split='train[:10]')

    # Loop over each episode in the dataset
    for episode_idx, episode in enumerate(dataset):
        print(f"\nProcessing episode {episode_idx + 1}")

        # Get images from the episode
        images = [step['observation']['image_0'] for step in episode['steps']]
        images = [Image.fromarray(image.numpy()) for image in images]
        print(f"Number of frames in episode: {len(images)}")

        # Create generated video (copy of robot video for now)
        generated_frames = images.copy()
        robot_frames = images.copy()

        # Load tracking results (assumes tracks are saved per episode)
        episode_label = construct_episode_label(episode)
        tracks_path = f"{bridge_data_path}/{episode_label}_tracks.npz"
        visibles_path = f"{bridge_data_path}/{episode_label}_visibles.npz"

        if not os.path.exists(tracks_path) or not os.path.exists(visibles_path):
            print(f"Tracking results not found for episode {episode_label}, skipping...")
            continue

        tracks, visibles = load_tracking_results(tracks_path, visibles_path)

        # Process generated video
        # Sample 16 frames, ensuring first and last frames are included
        total_frames_gen = len(generated_frames)
        sampled_indices_gen = get_sample_indices(total_frames_gen, num_frames_to_sample=16, last_frames=False)
        z_g, i0_g, P0_g, predicted_tracks_g, gt_tracks_g = process_video(
            generated_frames, tracks, sampled_indices_gen, device, video_type='generated'
        )

        # Compute auxiliary loss for generated video
        aux_loss_g = F.mse_loss(predicted_tracks_g, gt_tracks_g)
        print(f"Auxiliary loss for generated video: {aux_loss_g.item()}")

        # # Process robot video
        # # Get the last 8 frames
        total_frames_robot = len(robot_frames)
        sampled_indices_robot = get_sample_indices(total_frames_robot, num_frames_to_sample=8, last_frames=True)
        z_r, i0_r, P0_r, predicted_tracks_r, gt_tracks_r = process_video(
            robot_frames, tracks, sampled_indices_robot, device, video_type='robot'
        )

        # Compute auxiliary loss for robot video
        aux_loss_r = F.mse_loss(predicted_tracks_r, gt_tracks_r)
        print(f"Auxiliary loss for robot video: {aux_loss_r.item()}")

        # TODO: Combine auxiliary losses and proceed with behavior cloning loss
        # For now, we can simply sum them as per the paper (during training)
        total_aux_loss = aux_loss_g  + aux_loss_r
        print(f"Total auxiliary loss: {total_aux_loss.item()}")

        # Break after a few episodes for demonstration purposes
        if episode_idx >= 9:
            break

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run video processing with specified GPU.')
    parser.add_argument('--bridge_data_path', type=str, default="/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/",
                        help='Path to the bridge dataset directory.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use, e.g., cuda:0, cuda:1, or cpu (default: cuda:0)')
    args = parser.parse_args()

    main(bridge_data_path=args.bridge_data_path, device=args.device)