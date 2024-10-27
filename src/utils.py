from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import json

def count_episodes(dataset):
    """Counts the number of episodes in the dataset."""
    return sum(1 for _ in dataset)

def construct_episode_label(episode):
    episode_id = int(episode['episode_metadata']['episode_id'].numpy())
    episode_fp = episode['episode_metadata']['file_path'].numpy().decode('utf-8')
    episode_fp = str(Path(episode_fp).with_suffix(''))
    episode_fp_clean = episode_fp.replace("/", "_")
    episode_label = f"episode_{episode_id}___{episode_fp_clean}"
    
    return episode_label

def load_tracking_results(tracks_path, visibles_path):
    """
    Load tracking results from given file paths, with simplified error handling.

    Args:
        tracks_path (Path): Path to the .npz file containing tracks data.
        visibles_path (Path): Path to the .npz file containing visibles data.

    Returns:
        tuple: (tracks, visibles) loaded from the .npz files, or (None, None) if loading fails.
    """
    try:
        tracks = np.load(tracks_path, allow_pickle=True)['tracks']
    except Exception as e:
        print(f"Error loading tracks from {tracks_path}: {e}")
        tracks = None

    try:
        visibles = np.load(visibles_path, allow_pickle=True)['visibles']
    except Exception as e:
        print(f"Error loading visibles from {visibles_path}: {e}")
        visibles = None

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
        'config': tensor_to_primitive(dict(config)),
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

def compute_action_accuracies(pred_actions_discrete, gt_actions_discrete, mode='train'):
    """
    Compute various accuracy metrics for action predictions.
    
    Args:
        pred_actions_discrete: Predicted action tensor of shape (batch_size, num_future_actions, action_dim)
        gt_actions_discrete: Ground truth action tensor of the same shape as pred_actions_discrete

    Returns:
        dict: Dictionary containing accuracy metrics.
    """
    pos_idx = slice(0, 3)
    rot_idx = slice(3, 6)
    gripper_idx = 6

    batch_size, num_future_actions, _ = pred_actions_discrete.shape

    # Strict accuracy metrics
    correct_strict_first_action = (pred_actions_discrete[:, 0] == gt_actions_discrete[:, 0]).all(dim=1).sum().item()
    correct_strict_all_actions = (pred_actions_discrete == gt_actions_discrete).all(dim=2).all(dim=1).sum().item()

    # Per-component accuracy for first action
    pos_x_acc_first = (pred_actions_discrete[:, 0, 0] == gt_actions_discrete[:, 0, 0]).sum().item()
    pos_y_acc_first = (pred_actions_discrete[:, 0, 1] == gt_actions_discrete[:, 0, 1]).sum().item()
    pos_z_acc_first = (pred_actions_discrete[:, 0, 2] == gt_actions_discrete[:, 0, 2]).sum().item()

    rot_x_acc_first = (pred_actions_discrete[:, 0, 3] == gt_actions_discrete[:, 0, 3]).sum().item()
    rot_y_acc_first = (pred_actions_discrete[:, 0, 4] == gt_actions_discrete[:, 0, 4]).sum().item()
    rot_z_acc_first = (pred_actions_discrete[:, 0, 5] == gt_actions_discrete[:, 0, 5]).sum().item()

    gripper_acc_first = (pred_actions_discrete[:, 0, 6] == gt_actions_discrete[:, 0, 6]).sum().item()

    # Strict position and rotation accuracy for first action
    correct_position_first = (pred_actions_discrete[:, 0, pos_idx] == gt_actions_discrete[:, 0, pos_idx]).all(dim=1).sum().item()
    correct_rotation_first = (pred_actions_discrete[:, 0, rot_idx] == gt_actions_discrete[:, 0, rot_idx]).all(dim=1).sum().item()

    # Per-component accuracy across all future actions
    pos_x_acc_all = (pred_actions_discrete[:, :, 0] == gt_actions_discrete[:, :, 0]).sum().item()
    pos_y_acc_all = (pred_actions_discrete[:, :, 1] == gt_actions_discrete[:, :, 1]).sum().item()
    pos_z_acc_all = (pred_actions_discrete[:, :, 2] == gt_actions_discrete[:, :, 2]).sum().item()

    rot_x_acc_all = (pred_actions_discrete[:, :, 3] == gt_actions_discrete[:, :, 3]).sum().item()
    rot_y_acc_all = (pred_actions_discrete[:, :, 4] == gt_actions_discrete[:, :, 4]).sum().item()
    rot_z_acc_all = (pred_actions_discrete[:, :, 5] == gt_actions_discrete[:, :, 5]).sum().item()

    gripper_acc_all = (pred_actions_discrete[:, :, 6] == gt_actions_discrete[:, :, 6]).sum().item()

    # Strict position and rotation accuracy across all actions
    correct_position_all = (pred_actions_discrete[:, :, pos_idx] == gt_actions_discrete[:, :, pos_idx]).all(dim=2).all(dim=1).sum().item()
    correct_rotation_all = (pred_actions_discrete[:, :, rot_idx] == gt_actions_discrete[:, :, rot_idx]).all(dim=2).all(dim=1).sum().item()

    # Strict accuracy for gripper across all actions
    correct_gripper_all = (pred_actions_discrete[:, :, gripper_idx] == gt_actions_discrete[:, :, gripper_idx]).all(dim=1).sum().item()

    # Normalize metrics
    metrics = {
        'strict_first_action': correct_strict_first_action / batch_size,
        'strict_all_actions': correct_strict_all_actions / batch_size,
        'pos_x_acc_first': pos_x_acc_first / batch_size,
        'pos_y_acc_first': pos_y_acc_first / batch_size,
        'pos_z_acc_first': pos_z_acc_first / batch_size,
        'rot_x_acc_first': rot_x_acc_first / batch_size,
        'rot_y_acc_first': rot_y_acc_first / batch_size,
        'rot_z_acc_first': rot_z_acc_first / batch_size,
        'gripper_acc_first': gripper_acc_first / batch_size,
        'strict_position_first': correct_position_first / batch_size,
        'strict_rotation_first': correct_rotation_first / batch_size,
        'pos_x_acc_all': pos_x_acc_all / (batch_size * num_future_actions),
        'pos_y_acc_all': pos_y_acc_all / (batch_size * num_future_actions),
        'pos_z_acc_all': pos_z_acc_all / (batch_size * num_future_actions),
        'rot_x_acc_all': rot_x_acc_all / (batch_size * num_future_actions),
        'rot_y_acc_all': rot_y_acc_all / (batch_size * num_future_actions),
        'rot_z_acc_all': rot_z_acc_all / (batch_size * num_future_actions),
        'gripper_acc_all': gripper_acc_all / (batch_size * num_future_actions),
        'strict_position_all': correct_position_all / batch_size,
        'strict_rotation_all': correct_rotation_all / batch_size,
        'strict_gripper_all': correct_gripper_all / batch_size,
    }
    
    # Add mode prefix to each metric key
    # metrics = {f'{mode}_{key}': value for key, value in metrics.items()}

    return metrics


def test_compute_action_accuracies():
    batch_size = 2
    num_future_actions = 4
    action_dim = 7

    # Test case 1: Perfect match
    gt_actions_discrete = torch.randint(0, 256, (batch_size, num_future_actions, action_dim))
    pred_actions_discrete = gt_actions_discrete.clone()
    metrics = compute_action_accuracies(pred_actions_discrete, gt_actions_discrete)
    
    print("Metrics for perfect match test case:", metrics)

    assert metrics['strict_first_action'] == 1.0, "Failed perfect match - strict first action"
    assert metrics['strict_all_actions'] == 1.0, "Failed perfect match - strict all actions"
    assert all(value == 1.0 for key, value in metrics.items()), "Failed perfect match - any metric"

    # Test case 2: Complete mismatch
    pred_actions_discrete = (gt_actions_discrete + 1) % 256
    metrics = compute_action_accuracies(pred_actions_discrete, gt_actions_discrete)

    assert metrics['strict_first_action'] == 0.0, "Failed complete mismatch - strict first action"
    assert metrics['strict_all_actions'] == 0.0, "Failed complete mismatch - strict all actions"
    assert all(value == 0.0 for key, value in metrics.items()), "Failed complete mismatch - any metric"

    # Test case 3: Only position matches
    pred_actions_discrete = gt_actions_discrete.clone()
    pred_actions_discrete[:, :, 3:6] += 1  # Change rotation
    pred_actions_discrete[:, :, 6] += 1    # Change gripper
    metrics = compute_action_accuracies(pred_actions_discrete, gt_actions_discrete)

    assert metrics['pos_x_acc_first'] == 1.0, "Failed partial match - pos X first"
    assert metrics['pos_y_acc_first'] == 1.0, "Failed partial match - pos Y first"
    assert metrics['pos_z_acc_first'] == 1.0, "Failed partial match - pos Z first"
    assert metrics['strict_rotation_first'] == 0.0, "Failed partial match - strict rotation first"
    assert metrics['gripper_acc_first'] == 0.0, "Failed partial match - gripper first"
    assert metrics['strict_position_all'] == 1.0, "Failed partial match - strict position all"

    print("All test cases passed!")

# # Run the test
# test_compute_action_accuracies()