from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import json

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
        
def tensor_to_primitive(obj):
    """
    Recursively convert Tensors to lists and handle other non-serializable types.

    Args:
        obj: Any Python object.

    Returns:
        A JSON-serializable version of the object.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert Tensor to list
    elif isinstance(obj, dict):
        return {k: tensor_to_primitive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_primitive(v) for v in obj]
    else:
        return obj
        
        
def save_model(model, save_path, model_name):
    """
    Save model weights to a specified path.
    
    Args:
        model (nn.Module): Model to save.
        save_path (Path): Directory path to save the model.
        model_name (str): Name of the model.
    """
    save_file = save_path / f"{model_name}.pth"
    torch.save(model.state_dict(), save_file)
    print(f"Saved {model_name} to {save_file}")

def save_checkpoint(epoch_label, models, optimizer, config, run_logs, save_dir):
    """
    Save models, optimizer state, config, and run logs.
    
    Args:
        epoch_label (int): Current epoch number.
        models (dict): Dictionary of models to save.
        optimizer (Optimizer): Optimizer to save.
        config (dict): Configuration dictionary.
        run_logs (dict): Run logs dictionary.
        save_dir (str): Directory to save checkpoints.
    """
    # Create the main save directory based on epoch
    folder_name = f'epoch_{epoch_label}'
    save_path = Path(save_dir) / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save models individually
    for model_name, model in models.items():
        save_model(model, save_path, model_name)

    # Save config and run logs using JSON
    serializable_checkpoint = {
        'epoch': epoch_label,
        'config': tensor_to_primitive(config),
        'run_logs': tensor_to_primitive(run_logs)
    }
    json_path = save_path / 'checkpoint.json'
    with open(json_path, 'w') as f:
        json.dump(serializable_checkpoint, f)
    print(f"Serializable parts saved to {json_path}")

    # Save non-serializable optimizer state using torch.save
    torch_path = save_path / 'optimizer_state.pth'
    torch.save(optimizer.state_dict(), torch_path)
    print(f"Non-serializable optimizer state saved to {torch_path}")
    
def load_model_weights(model, model_path, device):
    """
    Helper function to load model weights with non-strict loading.

    Args:
        model (nn.Module): Model to load weights into.
        model_path (Path): Path to the model weights.
        device (str): Device to load the model onto.
    """
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found.")

def load_checkpoint(checkpoint_dir, device, models, optimizer):
    """
    Load models, optimizer state, config, and run logs from a checkpoint.
    
    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        device (str): Device to load the models onto.
        models (dict): Dictionary of models to load weights into.
        optimizer (Optimizer): Optimizer to load state into.
        
    Returns:
        tuple: config and run_logs from the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    print(f"Loading from checkpoint directory: {checkpoint_dir}")

    # Load models
    for model_name, model in models.items():
        model_path = checkpoint_dir / f"{model_name}.pth"
        load_model_weights(model, model_path, device)

    # Load config and run logs from JSON
    json_path = checkpoint_dir / 'checkpoint.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            checkpoint = json.load(f)
        config = checkpoint.get('config', {})
        run_logs = checkpoint.get('run_logs', {})
        print(f"Loaded config and run logs from {json_path}")
    else:
        print(f"Warning: Checkpoint JSON file {json_path} not found.")
        config, run_logs = {}, {}

    # Load optimizer state using torch.load
    torch_path = checkpoint_dir / 'optimizer_state.pth'
    if torch_path.exists():
        optimizer.load_state_dict(torch.load(torch_path, map_location=device))
        print(f"Loaded optimizer state from {torch_path}")
    else:
        print(f"Warning: Optimizer state file {torch_path} not found.")

    return config, run_logs