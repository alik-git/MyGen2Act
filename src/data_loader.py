# data_loader.py

import argparse
import torch

from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

# Enable dynamic memory growth, otherwise tensorflow hogs the whole dataset on the GPU immediately
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from src.utils import construct_episode_label
from src.utils import load_tracking_results

from src.pt_tracker import generate_tracking_files, load_tapir_model


def get_sample_indices(num_steps, num_samples=16):
    """
    Sample frame indices for a video based on the specified number of samples. 
    This matches the method in the Gen2Act paper

    Args:
    - num_steps (int): Total number of frames in the video.
    - num_samples (int): Number of frames to sample (default: 16).

    Returns:
    - episode_sampled_indices (list): List of sampled frame indices.
    """
    # Ensure we have at least 'num_samples' frames
    if num_steps >= num_samples:
        # Start with the first frame
        episode_sampled_indices = [0]

        # Sample middle frames if there are more than 2 frames
        if num_steps > 2:
            middle_indices = np.linspace(1, num_steps - 2, num_samples - 2, dtype=int)
            episode_sampled_indices.extend(middle_indices.tolist())

        # Include the last frame
        episode_sampled_indices.append(num_steps - 1)
    else:
        # If fewer frames than 'num_samples', use all available frames
        episode_sampled_indices = list(range(num_steps))

    return episode_sampled_indices

def process_episode(episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model=None):
    """
    Process a single episode to extract trajectories and related data.
    If tracking files are missing, generate them using the TAPIR model.

    Args:
    - episode (dict): Episode data containing multiple steps.
    - trajectory_length (int): Number of frames in each trajectory.
    - next_actions_length (int): Number of future actions to collect as labels.
    - tracks_dir (str): Directory path for point tracking data.
    - tapir_model (torch.nn.Module, optional): TAPIR model for generating tracking files.

    Returns:
    - List of trajectories extracted from the episode.
    - tapir_model (torch.nn.Module): TAPIR model (may be initialized during processing).
    """
    
    trajectories = []
    steps = list(episode['steps'].as_numpy_iterator())
    num_steps = len(steps)

    # Sample frames for the entire episode
    episode_sampled_indices = get_sample_indices(num_steps, num_samples=16)
    whole_episode_images = np.stack(
        [steps[idx]['observation']['image_0'] for idx in episode_sampled_indices], axis=0
    )

    episode_label = construct_episode_label(episode)
    
    # Define paths for tracking files
    tracks_path = Path(tracks_dir) / f"{episode_label}_tracks.npz"
    visibles_path = Path(tracks_dir) / f"{episode_label}_visibles.npz"
    
    # Check if tracking files are missing
    if not tracks_path.exists() or not visibles_path.exists():
        # Lazy initialization: Load the TAPIR model only if needed
        if tapir_model is None:
            print("Initializing TAPIR model for missing tracking files...")
            tapir_model = load_tapir_model(tapir_model_checkpoint_fp)

        # Generate tracking files using the TAPIR model
        generate_tracking_files(episode, tapir_model, tracks_dir)
        
    # Load tracking results
    episode_tracks, episode_visibles = load_tracking_results(tracks_path, visibles_path)


    for i in range(num_steps - trajectory_length + 1):
        # Extract trajectory steps and next actions
        trajectory_steps = steps[i:i + trajectory_length]
        next_actions_steps = steps[i + trajectory_length:i + trajectory_length + next_actions_length]

        # Prepare trajectory data
        trajectory_images = np.stack(
            [step['observation']['image_0'] for step in trajectory_steps], axis=0
        )
        trajectory_actions = np.stack(
            [step['action'] for step in trajectory_steps], axis=0
        )
        next_actions = np.zeros((next_actions_length, trajectory_actions.shape[1]), dtype=np.float32)
        for j, step in enumerate(next_actions_steps):
            next_actions[j] = step['action']

        trajectory_discount = np.stack(
            [step['discount'] for step in trajectory_steps], axis=0
        )
        trajectory_is_first = np.stack(
            [step['is_first'] for step in trajectory_steps], axis=0
        )
        trajectory_is_last = np.stack(
            [step['is_last'] for step in trajectory_steps], axis=0
        )
        trajectory_is_terminal = np.stack(
            [step['is_terminal'] for step in trajectory_steps], axis=0
        )
        language_instruction = trajectory_steps[0]['language_instruction']
        trajectory_reward = np.stack(
            [step['reward'] for step in trajectory_steps], axis=0
        )

        # # Load point tracking data
        # tracks_path = f"{tracks_dir}/{episode_label}_tracks.npz"
        # visibles_path = f"{tracks_dir}/{episode_label}_visibles.npz"
        
        # # Generate tracking files if they do not exist
        # if not tracks_path.exists() or not visibles_path.exists():
        #     generate_tracking_files(episode, tapir_model, tracks_dir)

        # tracks, visibles = load_tracking_results(tracks_path, visibles_path)

        trajectory_tracks = episode_tracks[:, i:i + trajectory_length, :]
        trajectory_visibles = episode_visibles[:, i:i + trajectory_length]
        whole_episode_tracks = episode_tracks[:, episode_sampled_indices, :]
        whole_episode_visibles = episode_visibles[:, episode_sampled_indices]

        trajectory = {
            'trajectory_images': trajectory_images,
            'trajectory_actions': trajectory_actions,
            'next_actions': next_actions,
            'trajectory_discount': trajectory_discount,
            'trajectory_is_first': trajectory_is_first,
            'trajectory_is_last': trajectory_is_last,
            'trajectory_is_terminal': trajectory_is_terminal,
            'language_instruction': language_instruction,
            'trajectory_reward': trajectory_reward,
            'whole_episode_images': whole_episode_images,
            'trajectory_tracks': trajectory_tracks,
            'trajectory_visibles': trajectory_visibles,
            'whole_episode_tracks': whole_episode_tracks,
            'whole_episode_visibles': whole_episode_visibles
        }
        trajectories.append(trajectory)

    return trajectories, tapir_model


def build_dataset(bridge_data_path, tracks_dir, tapir_model_checkpoint_fp, trajectory_length=8, next_actions_length=4, train_split='train', val_split='val', batch_size=1):
    """
    Builds a tf.data.Dataset with both training and validation splits.

    Args:
        bridge_data_path (str): Path to the bridge dataset directory.
        tracks_dir (str): Path to the directory containing point tracking data.
        trajectory_length (int): Number of frames in each trajectory.
        next_actions_length (int): Number of future actions to collect as labels.
        train_split (str): Split for training data.
        val_split (str): Split for validation data.
        batch_size (int): Batch size for the dataset.

    Returns:
        Tuple: (tf.data.Dataset for training, tf.data.Dataset for validation)
        
    Data Structure Shapes:

    'trajectory_images': (trajectory_length, 256, 256, 3)
        - Shape: (8, 256, 256, 3) - 8 frames, 256x256 pixels, 3 channels.

    'trajectory_actions': (trajectory_length, 7)
        - Shape: (8, 7) - 8 steps, 7 action dimensions.

    'next_actions': (next_actions_length, 7)
        - Shape: (4, 7) - 4 future actions, 7 dimensions.

    'trajectory_discount': (trajectory_length,)
        - Shape: (8,) - 8 discount values.

    'trajectory_is_first': (trajectory_length,)
        - Shape: (8,) - 8 boolean flags indicating first step.

    'trajectory_is_last': (trajectory_length,)
        - Shape: (8,) - 8 boolean flags indicating last step.

    'trajectory_is_terminal': (trajectory_length,)
        - Shape: (8,) - 8 boolean flags indicating terminal step.

    'language_instruction': scalar string
        - Shape: scalar string.

    'trajectory_reward': (trajectory_length,)
        - Shape: (8,) - 8 reward values.

    'whole_episode_images': (16, 256, 256, 3)
        - Shape: (16, 256, 256, 3) - 16 frames, 256x256 pixels, 3 channels.

    'trajectory_tracks': (1024, trajectory_length, 2)
        - Shape: (1024, 8, 2) - 1024 points, 8 frames, 2D coordinates.

    'trajectory_visibles': (1024, trajectory_length)
        - Shape: (1024, 8) - 1024 visibility flags, 8 frames.

    'whole_episode_tracks': (1024, 16, 2)
        - Shape: (1024, 16, 2) - 1024 points, 16 frames, 2D coordinates.

    'whole_episode_visibles': (1024, 16)
        - Shape: (1024, 16) - 1024 visibility flags, 16 frames.
        
    """
    
    print ("Building dataset")
    
    # Initialize TAPIR model as None; will be loaded only if needed
    tapir_model = None
    
    # Load the dataset
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    
    # # Print available splits
    # print("Available splits:")
    # for split_name, split_info in dataset_builder.info.splits.items():
    #     num_examples = split_info.num_examples
    #     print(f"- {split_name}: {num_examples} examples")
        
        
    # Load train and validation splits
    train_dataset = dataset_builder.as_dataset(split=train_split)
    val_dataset = dataset_builder.as_dataset(split=val_split)

    train_trajectories = []
    for episode in train_dataset:
        curr_train_trajectories, tapir_model = process_episode(
            episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model
        )
        train_trajectories.extend(curr_train_trajectories)

    val_trajectories = []
    for episode in val_dataset:
        curr_val_trajectories, tapir_model = process_episode(
            episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model
        )
        val_trajectories.extend(curr_val_trajectories)

    # Create a tf.data.Dataset from trajectories
    def gen(data):
        for traj in data:
            yield traj

    # Define the output types and shapes based on the first trajectory
    sample_trajectory = train_trajectories[0]
    output_types = {
        'trajectory_images': tf.uint8,
        'trajectory_actions': tf.float32,
        'next_actions': tf.float32,
        'trajectory_discount': tf.float32,
        'trajectory_is_first': tf.bool,
        'trajectory_is_last': tf.bool,
        'trajectory_is_terminal': tf.bool,
        'language_instruction': tf.string,
        'trajectory_reward': tf.float32,
        'whole_episode_images': tf.uint8,
        'trajectory_tracks': tf.float64,
        'trajectory_visibles': tf.bool,
        'whole_episode_tracks': tf.float64,
        'whole_episode_visibles': tf.bool
    }
    output_shapes = {}
    for key, value in sample_trajectory.items():
        if isinstance(value, bytes):  # Handle string separately
            output_shapes[key] = tf.TensorShape([])  # Scalar shape for strings
        else:
            output_shapes[key] = tf.TensorShape(value.shape)  # Regular shape for tensors/arrays


    # Create tf.data.Dataset for training and validation
    train_tf_dataset = tf.data.Dataset.from_generator(
        lambda: gen(train_trajectories),
        output_types=output_types,
        output_shapes=output_shapes
    ).batch(batch_size)

    val_tf_dataset = tf.data.Dataset.from_generator(
        lambda: gen(val_trajectories),
        output_types=output_types,
        output_shapes=output_shapes
    ).batch(batch_size)

    print("Training and validation datasets built successfully.")
    return train_tf_dataset, val_tf_dataset


def process_data_sample(data_sample):
    """
    Dummy processing function for a dataset sample.
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset with point tracking data.')
    parser.add_argument('--bridge_data_path', type=str, required=True, help='Path to the bridge dataset directory.')
    parser.add_argument('--tracks_dir', type=str, required=True, help='Path to the directory containing point tracks.')
    parser.add_argument('--tapir_model_checkpoint_fp', type=str, required=True, help='Path to the TAPIR model checkpoint.')

    args = parser.parse_args()

    # Build the dataset
    train_dataset, val_dataset = build_dataset(args.bridge_data_path, args.tracks_dir, args.tapir_model_checkpoint_fp)

    # Iterate through the dataset and process each sample
    for train_sample in train_dataset.take(5):  # Limit to 5 samples for demonstration
        process_data_sample(train_sample)
        print("Processed a train_data sample.")
        
    # Iterate through the dataset and process each sample
    for val_sample in val_dataset.take(5):  # Limit to 5 samples for demonstration
        process_data_sample(val_sample)
        print("Processed a val_data sample.")
