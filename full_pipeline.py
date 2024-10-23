# full_pipeline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import tensorflow as tf

import os

import argparse
from pathlib import Path

# Import build_dataset from data_loader.py
from src.action_prediction_transformer import ActionPredictionTransformer
from src.data_loader import build_dataset

from src.vit_ft_extractor import VideoFeatureExtractor
from src.perceiver_resampler import PerceiverResampler
from src.track_pred_transformer import TrackPredictionTransformer

# Helper functions
def preprocess_frames(frames, device, resize_height=224, resize_width=224):
    """
    Preprocess frames: resize and normalize.
    frames: numpy array of shape (batch_size, num_frames, H, W, C)
    Returns: torch tensor of shape (batch_size, num_frames, 3, resize_height, resize_width)
    """
    batch_size, num_frames, H, W, C = frames.shape

    # Flatten batch and time dimensions
    frames = frames.reshape(-1, H, W, C)  # shape: (batch_size * num_frames, H, W, C)

    # Convert to torch tensor and normalize to [0, 1]
    frames = torch.from_numpy(frames).float().div(255).to(device)  # shape: (batch_size * num_frames, H, W, C)

    # Permute to (batch_size * num_frames, C, H, W)
    frames = frames.permute(0, 3, 1, 2)  # shape: (batch_size * num_frames, C, H, W)

    # Resize
    frames = torch.nn.functional.interpolate(frames, size=(resize_height, resize_width), mode='bilinear', align_corners=False)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # Reshape back to (batch_size, num_frames, 3, resize_height, resize_width)
    frames = frames.view(batch_size, num_frames, 3, resize_height, resize_width)

    return frames  # torch tensor

def process_video(frames, tracks, device, feature_extractor, perceiver_resampler, gen_video_track_predictor, robot_video_track_predictor, action_predictor, video_type='generated'):
    """
    Process a batch of videos through ViT, Perceiver Resampler, and Track Prediction Transformer.
    Returns the latent tokens, initial ViT features, initial points, and predicted tracks.
    frames: numpy array of shape (batch_size, num_frames, H, W, C)
    tracks: numpy array of shape (batch_size, num_points, num_frames, 2)
    """
    batch_size, num_frames_processed, H, W, C = frames.shape
    num_points = tracks.shape[1]

    # Preprocess frames
    processed_frames = preprocess_frames(frames, device)  # shape: (batch_size, num_frames, 3, H', W')
    print(f"Processed {video_type} video frames: {processed_frames.shape}")

    # Extract features from video frames
    features = feature_extractor.extract_features(processed_frames)
    # features: shape (batch_size, num_frames, num_tokens, hidden_dim)
    print(f"{video_type.capitalize()} video features shape: {features.shape}")

    # Extract initial ViT features (from the first frame)
    i0 = features[:, 0, :, :]  # Shape: [batch_size, num_tokens, hidden_dim]

    # Flatten features for PerceiverResampler
    batch_size, num_frames_processed, num_tokens, hidden_dim = features.shape
    seq_len = num_frames_processed * num_tokens
    features_flat = features.view(batch_size, -1, hidden_dim)  # Flatten frames and tokens


    # Pass features through PerceiverResampler
    z = perceiver_resampler(features_flat)  # Shape: [batch_size, num_latents, hidden_dim]
    print(f"{video_type.capitalize()} video latent tokens shape: {z.shape}")

    # Prepare initial points P0
    P0 = tracks[:, :, 0, :]  # Shape: [batch_size, num_points, 2]

    # Normalize coordinates to be between 0 and 1
    img_height, img_width = frames.shape[2], frames.shape[3]
    P0_normalized = P0 / np.array([img_width, img_height])

    # Convert to PyTorch tensor
    P0_normalized_tensor = torch.tensor(P0_normalized, device=device).float()

    # Prepare ground truth tracks τ
    gt_tracks_normalized = tracks / np.array([img_width, img_height])
    gt_tracks_normalized_tensor = torch.tensor(gt_tracks_normalized, device=device).float()

    # Predict tracks τ̂
    predicted_tracks = None
    if video_type == 'generated':       
        predicted_tracks = gen_video_track_predictor(P0_normalized_tensor, i0, z)  # [batch_size, num_points, num_frames, 2]
    elif video_type == 'robot':
        predicted_tracks = robot_video_track_predictor(P0_normalized_tensor, i0, z)

    return z, i0, P0_normalized_tensor, predicted_tracks, gt_tracks_normalized_tensor

def main(bridge_data_path, tracks_dir, device='cuda:0', batch_size=1):
    """
    Main function to build the dataset and run the video processing pipeline.

    Args:
        bridge_data_path (str): Path to the bridge dataset directory.
        tracks_dir (str): Path to the point tracks directory.
        device (str): Device to use (e.g., cuda:0, cuda:1, or cpu).
        batch_size (int): Batch size for processing trajectories.
    """
    print("==== Starting Video Processing Pipeline ====")
    print(f"Bridge Data Path: {bridge_data_path}")
    print(f"Tracks Directory: {tracks_dir}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")

    # Build the dataset
    trajectories = build_dataset(
        bridge_data_path=bridge_data_path,
        tracks_dir=tracks_dir,
        trajectory_length=8,
        next_actions_length=4,
        split='train[:10]',
        batch_size=batch_size
    )

    # Initialize models
    feature_extractor = VideoFeatureExtractor(device=device)
    perceiver_resampler = PerceiverResampler(
        dim=768, num_latents=64, depth=2, heads=8, dim_head=64, ff_mult=4
    ).to(device)

    gen_video_track_predictor = TrackPredictionTransformer(
        point_dim=2, hidden_dim=768, num_layers=6, num_heads=8, num_frames=16
    ).to(device)
    robot_video_track_predictor = TrackPredictionTransformer(
        point_dim=2, hidden_dim=768, num_layers=6, num_heads=8, num_frames=8
    ).to(device)

    # Initialize ActionPredictionTransformer
    action_predictor = ActionPredictionTransformer(
        hidden_dim=768,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        action_dim=7,  # As per your action dimensions
        num_future_actions=4,  # Predict next 4 actions
        num_bins=256,
        dropout=0.1
    ).to(device)

    # Loss function for action prediction
    action_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(action_predictor.parameters(), lr=1e-4)

    # Iterate through the batches
    for batch_idx, batch in enumerate(trajectories):
        print(f"\nProcessing batch {batch_idx + 1}")

        # Get images from the batch
        # For generated video (whole_episode_images)
        generated_frames = batch['whole_episode_images'].numpy()  # shape: (batch_size, num_frames_gen, H, W, C)
        # For robot video (trajectory_images)
        robot_frames = batch['trajectory_images'].numpy()  # shape: (batch_size, num_frames_robot, H, W, C)

        # Get tracks
        tracks_whole_episode = batch['whole_episode_tracks'].numpy()  # shape: (batch_size, num_points, num_frames_gen, 2)
        tracks_robot = batch['trajectory_tracks'].numpy()  # shape: (batch_size, num_points, num_frames_robot, 2)

        # Process generated video
        z_g, i0_g, P0_g, predicted_tracks_g, gt_tracks_g = (
            process_video(
                generated_frames,
                tracks_whole_episode,
                device,
                feature_extractor,
                perceiver_resampler,
                gen_video_track_predictor,
                robot_video_track_predictor,
                action_predictor,
                video_type="generated",
            )
        )

        # Compute auxiliary loss for generated video
        aux_loss_g = F.mse_loss(predicted_tracks_g, gt_tracks_g)
        print(f"Auxiliary loss for generated video: {aux_loss_g.item()}")

        # Process robot video
        z_r, i0_r, P0_r, predicted_tracks_r, gt_tracks_r = (
            process_video(
                robot_frames,
                tracks_robot,
                device,
                feature_extractor,
                perceiver_resampler,
                gen_video_track_predictor,
                robot_video_track_predictor,
                action_predictor,
                video_type="robot",
            )
        )

        # Compute auxiliary loss for robot video
        aux_loss_r = F.mse_loss(predicted_tracks_r, gt_tracks_r)
        print(f"Auxiliary loss for robot video: {aux_loss_r.item()}")

        # Combine auxiliary losses
        total_aux_loss = 0
        total_aux_loss += aux_loss_g
        total_aux_loss += aux_loss_r
        print(f"Total auxiliary loss: {total_aux_loss.item()}")
        
        # for now initialize z_g and z_r with random values
        
        
        
        # Predict actions
        action_logits = action_predictor(z_g, z_r)  # [batch_size, num_future_actions, action_dim, num_bins]

        # Ground truth actions
        gt_actions = batch['next_actions'].numpy()  # [batch_size, num_future_actions, action_dim]
        gt_actions = torch.tensor(gt_actions, device=device).float()

        # Discretize actions
        action_min = torch.tensor([-0.1791, -0.2553, -0.0681, -0.4487, -0.3450, -6.2708, 0.0], device=device)
        action_max = torch.tensor([0.1117, 0.2004, 0.1396, 0.5934, 0.3536, 6.2598, 1.0], device=device)

        gt_actions_normalized = (gt_actions - action_min) / (action_max - action_min)
        gt_actions_normalized = torch.clamp(gt_actions_normalized, 0.0, 1.0)
        gt_actions_discrete = (gt_actions_normalized * (action_predictor.num_bins - 1)).long()

        # Compute action prediction loss
        action_logits_flat = action_logits.view(-1, action_predictor.num_bins)
        gt_actions_flat = gt_actions_discrete.view(-1)
        action_loss = action_criterion(action_logits_flat, gt_actions_flat)

        # Total loss
        total_loss = total_aux_loss + action_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1} - Total Loss: {total_loss.item():.4f}, Action Loss: {action_loss.item():.4f}")


        # Break after a few batches for demonstration purposes
        if batch_idx >= 9:
            break

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run video processing with specified GPU and batch size.')
    parser.add_argument('--bridge_data_path', type=str, default="/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/",
                        help='Path to the bridge dataset directory.')
    parser.add_argument('--tracks_dir', type=str, default="/home/alik/MyGen2Act/local_data/point_tracks/",
                        help='Path to the point tracks directory.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use, e.g., cuda:0, cuda:1, or cpu (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing trajectories (default: 1)')
    args = parser.parse_args()

    main(bridge_data_path=args.bridge_data_path, tracks_dir=args.tracks_dir, device=args.device, batch_size=args.batch_size)
