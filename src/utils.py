from pathlib import Path
import numpy as np
import torch


def construct_episode_label(episode):
    episode_id = int(episode['episode_metadata']['episode_id'].numpy())
    episode_fp = episode['episode_metadata']['file_path'].numpy().decode('utf-8')
    episode_fp = str(Path(episode_fp).with_suffix(''))
    episode_fp_clean = episode_fp.replace("/", "_")
    episode_label = f"episode_{episode_id}___{episode_fp_clean}"
    
    return episode_label

def load_tracking_results(tracks_path, visibles_path):
    """
    Load tracks and visibles from .npz files.
    """
    tracks = np.load(tracks_path)['tracks']
    visibles = np.load(visibles_path)['visibles']
    # print(f"Tracks loaded from: {tracks_path}")
    # print(f"Visibles loaded from: {visibles_path}")
    return tracks, visibles


def get_sample_indices(total_frames, num_frames_to_sample, last_frames=False):
    """
    Returns sampled indices for a given total number of frames.

    Args:
    - total_frames: Total number of frames in the video.
    - num_frames_to_sample: Number of frames to sample.
    - last_frames: If True, samples the last n frames; otherwise, samples evenly with first and last included.

    Returns:
    - sampled_indices: List of sampled frame indices.
    """
    if total_frames <= 0 or num_frames_to_sample <= 0:
        raise ValueError("Both total_frames and num_frames_to_sample must be positive integers.")

    # Ensure the number of frames to sample does not exceed total frames
    num_frames_to_sample = min(num_frames_to_sample, total_frames)

    if last_frames:
        # Select the last num_frames_to_sample frames
        sampled_indices = list(range(total_frames - num_frames_to_sample, total_frames))
    else:
        # Evenly sample frames, ensuring the first and last frames are always included
        if num_frames_to_sample >= total_frames:
            sampled_indices = list(range(total_frames))  # Use all frames if fewer than num_frames_to_sample
        else:
            sampled_indices = [0]  # Start with the first frame
            step = (total_frames - 1) / (num_frames_to_sample - 1)
            for i in range(1, num_frames_to_sample - 1):
                sampled_indices.append(round(i * step))
            sampled_indices.append(total_frames - 1)  # Ensure the last frame is included

        # Remove duplicates and sort the indices
        sampled_indices = sorted(set(sampled_indices))

    return sampled_indices

def discretize_actions(actions, action_bins):
    """
    Discretize continuous actions into bins.

    Args:
        actions: [batch_size, num_actions, action_dim] - Continuous actions.
        action_bins: [action_dim, num_bins] - Bins for each action dimension.

    Returns:
        discretized_actions: [batch_size, num_actions, action_dim] - Discrete action indices.
    """
    batch_size, num_actions, action_dim = actions.shape
    discretized_actions = np.zeros((batch_size, num_actions, action_dim), dtype=np.int64)

    for d in range(action_dim):
        # For each action dimension, digitize the actions
        bins = action_bins[d]
        discretized_actions[:, :, d] = np.digitize(actions[:, :, d], bins) - 1  # Subtract 1 to get zero-based indices

    return discretized_actions

def create_action_bins(action_space_low, action_space_high, num_bins):
    """
    Create bins for discretizing actions.

    Args:
        action_space_low: [action_dim,] - Lower bounds of action dimensions.
        action_space_high: [action_dim,] - Upper bounds of action dimensions.
        num_bins: int - Number of bins per dimension.

    Returns:
        action_bins: [action_dim, num_bins] - Bins for each action dimension.
    """
    action_bins = []
    for low, high in zip(action_space_low, action_space_high):
        bins = np.linspace(low, high, num_bins + 1)[1:-1]  # Exclude the first and last edges
        action_bins.append(bins)
    return np.array(action_bins)

def track_memory(prefix, device='cuda:0', long=False, track_flag = False):
    """Helper function to print memory summary after synchronization.

    Args:
        prefix (str): Descriptive prefix for the memory summary.
        device (str): Device to check memory usage for (default is 'cuda:0').
        long (bool): If True, prints the full memory summary; if False, prints only allocated memory. 
    """
    if not track_flag:
        return
    torch.cuda.synchronize(device)  # Synchronize to ensure all operations are complete
    if long:
        print(f"\n==== {prefix} ====")
        print(torch.cuda.memory_summary(device=device))
    else:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"{prefix}: max allocated {allocated:.2f} MB")
        print(f"{prefix}: max reserved {max_reserved:.2f} MiB")