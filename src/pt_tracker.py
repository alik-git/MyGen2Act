import argparse
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F

from tapnet.torch import tapir_model
from tapnet.utils import transforms

from src.utils import construct_episode_label

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Define necessary functions
def preprocess_frames(frames):
    """Preprocess frames to model inputs."""
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames

def sample_grid_points(height, width, num_points_each_side):
    """Sample grid points with (time, height, width) order."""
    y_step = max(1, height // num_points_each_side)
    x_step = max(1, width // num_points_each_side)
    y = np.arange(0, height, y_step)
    x = np.arange(0, width, x_step)
    t = np.array([0])  # Only one time point (first frame)
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')
    points = np.stack([tt.ravel(), yy.ravel(), xx.ravel()], axis=-1).astype(np.int32)
    return points

def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5
    return visibles

def inference(frames, query_points, model):
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks = outputs['tracks'][0]  # [num_points, num_frames, 2]
    occlusions = outputs['occlusion'][0]
    expected_dist = outputs['expected_dist'][0]

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles

def save_tracking_results(tracks, visibles, tracks_output_fp, visibles_output_fp):
    """Save tracks and visibles to compressed .npz files."""
    np.savez_compressed(tracks_output_fp, tracks=tracks)
    print(f"Tracks saved to: {tracks_output_fp}")

    np.savez_compressed(visibles_output_fp, visibles=visibles)
    print(f"Visibles saved to: {visibles_output_fp}")

def generate_tracking_files(episode, model, output_dir, num_points_each_side=30):
    """
    Generate ground truth tracking files for a given episode.

    Args:
    - episode (dict): Episode data containing multiple steps.
    - model (torch.nn.Module): TAPIR model for tracking.
    - output_dir (str): Directory to save tracking results.
    - num_points_each_side (int): Number of points per side for sampling.
    """
    episode_label = construct_episode_label(episode)

    # Set the output file paths in the specified output directory
    tracks_output_fp = Path(output_dir) / f"{episode_label}_tracks.npz"
    visibles_output_fp = Path(output_dir) / f"{episode_label}_visibles.npz"

    # Check if tracking files already exist
    if tracks_output_fp.exists() and visibles_output_fp.exists():
        print(f"Tracking files already exist for: {episode_label}")
        return

    # Get images from the episode
    images = [step['observation']['image_0'] for step in episode['steps']]
    frames = np.stack([np.array(Image.fromarray(image.numpy())) for image in images], axis=0)

    # Sample grid points
    height, width = frames.shape[1:3]
    query_points = sample_grid_points(height, width, num_points_each_side)

    # Convert frames and query points to PyTorch tensors and move to device
    frames_tensor = torch.tensor(frames).to(device)
    query_points_tensor = torch.tensor(query_points).to(device)

    # Run inference
    tracks, visibles = inference(frames_tensor, query_points_tensor, model)

    # Convert tracks and visibles to NumPy arrays
    tracks = tracks.cpu().detach().numpy()
    visibles = visibles.cpu().detach().numpy()

    # Convert grid coordinates to pixel coordinates
    tracks = transforms.convert_grid_coordinates(tracks, (width, height), (width, height))

    # Save the tracking results
    save_tracking_results(tracks, visibles, tracks_output_fp, visibles_output_fp)
    
def load_tapir_model(model_checkpoint_fp):
    """
    Load the TAPIR model for inference.
    """
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(model_checkpoint_fp, map_location=device))
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def main(bridge_data_path, model_checkpoint_fp, output_dir, split='train[0:10]'):
    """
    Main function to run point tracking on the dataset.

    Args:
    - bridge_data_path (str): Path to the bridge dataset directory.
    - model_checkpoint_fp (str): Path to the model checkpoint file.
    - output_dir (str): Directory to save the point tracking results.
    - split (str): Dataset split to use for processing (default: 'train[0:10]').
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the TAPIR model
    model = load_tapir_model(model_checkpoint_fp)

    # Load the dataset
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    dataset = dataset_builder.as_dataset(split=split)

    # Iterate over episodes and generate tracking files
    for episode in dataset:
        generate_tracking_files(episode, model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TAPIR model inference on bridge dataset.')
    parser.add_argument('--bridge_data_path', type=str, required=True,
                        help='Path to the bridge dataset directory.')
    parser.add_argument('--model_checkpoint_fp', type=str, required=True,
                        help='Path to the model checkpoint file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the point tracking results.')
    parser.add_argument('--split', type=str, default='train[0:10]',
                        help='Dataset split to use for processing (default: "train[0:10]")')
    args = parser.parse_args()

    main(args.bridge_data_path, args.model_checkpoint_fp, args.output_dir, args.split)
