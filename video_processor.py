import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import mediapy as media
import numpy as np

import argparse

from pathlib import Path

from src.vit_ft_extractor import VideoFeatureExtractor
from src.perceiver_resampler import PerceiverResampler
from src.track_pred_transformer import TrackPredictionTransformer

# Helper functions
def preprocess_video(og_video, resize_height=224, resize_width=224):
    resized_video = media.resize_video(og_video, (resize_height, resize_width))
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resized_video_frames = [transform(frame).permute(1, 2, 0).numpy() for frame in resized_video]
    return resized_video, resized_video_frames

def load_tracking_results(video_path, load_tracks=True, load_visibles=True):
    """Load tracks and visibles from compressed .npz files."""
    video_path = Path(video_path)
    results = {}

    # Load tracks
    if load_tracks:
        load_path_tracks = video_path.with_stem(f"{video_path.stem}_tracks").with_suffix('.npz')
        results['tracks'] = np.load(load_path_tracks)['tracks']
        print(f"Tracks loaded from: {load_path_tracks}")

    # Load visibles
    if load_visibles:
        load_path_visibles = video_path.with_stem(f"{video_path.stem}_visibles").with_suffix('.npz')
        results['visibles'] = np.load(load_path_visibles)['visibles']
        print(f"Visibles loaded from: {load_path_visibles}")
    return results

def main(video_path, resize_height=224, resize_width=224, device='cuda:0'):
    
    # Get original video shape
    og_video = media.read_video(video_path)
    og_video_height, og_video_width = og_video.shape[1:3]
    print(f"Video frames shape: (num_frames, height, width, channels) = {og_video.shape}")


    # Load and preprocess video frames
    resized_video, resized_video_frames = preprocess_video(og_video, resize_height, resize_width)
    print(f"Resized video frames shape: (num_frames, height, width, channels) = {resized_video.shape}")

    
    # Initialize the ViT feature extractor 
    feature_extractor = VideoFeatureExtractor(device=device)
    # Extract features from video frames (features are basically i_g or i_r from the paper)
    # sampled_indices contains the indices of frames used, which we need to sync w pt tracks
    
    # Extract features from video frames (features are i_g or i_r from the paper)
    features, sampled_indices = feature_extractor.extract_features(resized_video_frames, num_frames_to_process=16)
    print("Extracted features shape: (num_frames_processed, num_tokens, hidden_dim) = ", features.shape)
    
    # this is needed for the unsqueeze below
    features = torch.tensor(features, device=device, dtype=torch.float32)

    # Extract i0_g (features from the first frame)
    i0_g = features[0].unsqueeze(0)  # Shape: [1, num_tokens, hidden_dim]
    print("i0_g shape: (1, num_tokens, hidden_dim) = ", i0_g.shape)

    # Flatten features for PerceiverResampler
    num_frames_processed, num_tokens, hidden_dim = features.shape
    seq_len = num_frames_processed * num_tokens
    features_flat = features.view(1, seq_len, hidden_dim)  # Shape: [1, seq_len, hidden_dim]

    # Initialize PerceiverResampler
    perceiver_resampler = PerceiverResampler(dim=hidden_dim, num_latents=64, num_layers=2, num_heads=8, dim_head=64, ff_mult=4).to(device)

    # Pass features through PerceiverResampler
    latents = perceiver_resampler(features_flat)  
    print("PerceiverResampler output shape: (1, num_latents, hidden_dim) = ", latents.shape)

    # Now latents is zg from the paper
    zg = latents  # Shape: [1, 64, hidden_dim]
    print("zg shape: (1, num_latents, hidden_dim) = ", zg.shape)


    # Load preproccessed point tracking results  
    tracking_results = load_tracking_results(video_path)
    tracks = tracking_results['tracks']
    visibles = tracking_results['visibles']
    print(f"Tracks shape: {tracks.shape}")
    print(f"Visibles shape: {visibles.shape}")

    
    # Prepare initial points P0
    num_points = tracks.shape[0]  # Number of points
    P0 = tracks[:, 0, :]  # Shape: [num_points, 2] # tracks at frame 0
    # Normalize coordinates to be between 0 and 1 # Not sure if this is needed but ChatGPT said so
    img_height, img_width = og_video.shape[1:3]
    P0_normalized = P0 / np.array([img_width, img_height])
    P0_normalized = torch.tensor(P0_normalized, device=device).float().unsqueeze(0)  # Shape: [1, num_points, 2]
    print("P0_normalized shape: (1, num_points, 2) = ", P0_normalized.shape)

    # Prepare ground truth tracks τg
    # Select frames corresponding to sampled_indices
    sampled_indices = sorted(set(sampled_indices))
    gt_tracks = tracks[:, sampled_indices, :]  # Shape: [num_points, num_frames_processed, 2]
    # print (gt_tracks.shape)
    print("Ground truth tracks shape: (num_points, num_frames_processed, 2) = ", gt_tracks.shape)
    # Normalize coordinates
    gt_tracks_normalized = gt_tracks / np.array([img_width, img_height])
    gt_tracks_normalized = torch.tensor(gt_tracks_normalized, device=device).float().unsqueeze(0)  # Shape: [1, num_points, num_frames_processed, 2]

    # Initialize TrackPredictionTransformer
    track_predictor = TrackPredictionTransformer(point_dim=2, hidden_dim=hidden_dim, num_layers=6, num_heads=8, num_frames=num_frames_processed).to(device) # hyperparameters from the paper

    # Predict tracks τ̂_g
    predicted_tracks = track_predictor(P0_normalized, i0_g, zg)  # Shape: [1, num_points, num_frames_processed, 2]

    # Compute auxiliary loss L_τ = ||τg - τ̂g||^2
    aux_loss = F.mse_loss(predicted_tracks, gt_tracks_normalized)
    print("Auxiliary loss:", aux_loss.item())
    


    # this part is for vizualization only so commenting out for now 
    # Visualize and save the video with tracks
    # tracks, input_shape, output_shape
    # tracks = transforms.convert_grid_coordinates(tracks, (og_video_width, og_video_height),(resize_width, resize_height))
    # video_viz = viz_utils.paint_point_track(resized_video, tracks, visibles)
    # save_video_viz_path = f"{Path(video_path).with_suffix('')}_w_tracks_viz.mp4"
    # media.write_video(save_video_viz_path, video_viz, fps=10)
    # print(f"Video with tracked points saved to {save_video_viz_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run video processing with specified GPU.')
    parser.add_argument('--video_path', type=str, 
                        # default='/home/kasm-user/Uploads/Hand_picks_up_the_can.mp4', 
                        default='/home/kasm-user/SimplerEnv-OpenVLA/octo_policy_video2.mp4', 
                        help='Path to the input video file (default: current video path)')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use, e.g., cuda:0, cuda:1, or cpu (default: cuda:0)')
    args = parser.parse_args()




    main(video_path=args.video_path, device=args.device)
