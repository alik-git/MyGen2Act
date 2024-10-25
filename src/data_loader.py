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

import tqdm

from src.pt_tracker import generate_tracking_files, load_tapir_model

from src.utils import count_episodes


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
        
    # assert that there are at least num_samples frames
    assert len(episode_sampled_indices) >= num_samples, f"Number of sampled indices is less than {num_samples}"

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
    
    # Skip episodes with fewer than 16 frames
    if num_steps < 16:
        return trajectories, tapir_model

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



# Helper function to determine output types and shapes from a sample trajectory
def get_output_types_shapes(sample_trajectory):
    """
    Determines the output types and shapes from a sample trajectory.

    Args:
        sample_trajectory (dict): A sample trajectory from the dataset.

    Returns:
        output_types (dict): Dictionary of output types.
        output_shapes (dict): Dictionary of output shapes.
    """
    output_types = {}
    output_shapes = {}

    for key, value in sample_trajectory.items():
        if isinstance(value, bytes):  # Handle string separately
            output_types[key] = tf.string
            output_shapes[key] = tf.TensorShape([])  # Scalar shape for strings
        else:
            output_types[key] = tf.as_dtype(value.dtype)
            output_shapes[key] = tf.TensorShape(value.shape)

    return output_types, output_shapes

# Helper function to update running average
def update_running_average(current_avg, new_value, count):
    """
    Update the running average with a new value.

    Args:
        current_avg (float): The current average.
        new_value (int): The new value to include.
        count (int): The current count of values.

    Returns:
        new_avg (float): The updated average.
    """
    return (current_avg * (count - 1) + new_value) / count


# Generator to yield trajectories one-by-one as they are processed
def lazy_loader_gen(episodes, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp):
    """
    Generator to yield trajectories one-by-one as they are processed.
    """
    tapir_model = None  # Initialize the TAPIR model only if needed
    total_trajectories = 0
    episode_count = 0
    running_avg_trajectories = 0

    for episode in episodes:
        curr_trajectories, tapir_model = process_episode(
            episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model
        )
        episode_trajectories = len(curr_trajectories)
        total_trajectories += episode_trajectories
        episode_count += 1

        # Update the running average
        running_avg_trajectories = update_running_average(running_avg_trajectories, episode_trajectories, episode_count)

        # Log the running average for debugging (optional)
        print(f"\nEpisode {episode_count}: {episode_trajectories} trajectories, Running Avg: {running_avg_trajectories:.2f}\n")

        for traj in curr_trajectories:
            yield traj  # Yield each trajectory as it's processed


def build_dataset(bridge_data_path, tracks_dir, tapir_model_checkpoint_fp, trajectory_length=8, next_actions_length=4, train_split='train', val_split='val', batch_size=1, lazy_loading=True, shuffle=True, shuffle_buffer_size=10000):
    """
    Builds a tf.data.Dataset with both training and validation splits, with lazy loading option.

    Args:
        bridge_data_path (str): Path to the bridge dataset directory.
        tracks_dir (str): Path to the directory containing point tracking data.
        trajectory_length (int): Number of frames in each trajectory.
        next_actions_length (int): Number of future actions to collect as labels.
        train_split (str): Split for training data.
        val_split (str): Split for validation data.
        batch_size (int): Batch size for the dataset.
        lazy_loading (bool): Whether to use lazy loading (default: True).

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

    num_train_dataset_episodes = count_episodes(train_dataset)
    num_val_dataset_episodes = count_episodes(val_dataset)

    print(f"Number of episodes in the training dataset: {num_train_dataset_episodes}")
    print(f"Number of episodes in the validation dataset: {num_val_dataset_episodes}")
    
    # Shuffle before batching if needed
    if shuffle:
        print("Shuffling the dataset!!")
        train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
        val_dataset = val_dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    # Initialize counters for running average computation
    total_trajectories = 0
    total_episodes = 0
    
    
    # Get a sample trajectory from the generator to determine output types and shapes
    sample_episode = next(iter(train_dataset))
    sample_trajectories, _ = process_episode(
        sample_episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, None
    )
    output_types, output_shapes = get_output_types_shapes(sample_trajectories[0])

    if lazy_loading:
        print("Using lazy loading for the dataset.")
        train_tf_dataset = tf.data.Dataset.from_generator(
            lambda: lazy_loader_gen(train_dataset, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp),
            output_types=output_types,
            output_shapes=output_shapes
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_tf_dataset = tf.data.Dataset.from_generator(
            lambda: lazy_loader_gen(val_dataset, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp),
            output_types=output_types,
            output_shapes=output_shapes
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print("Built lazy loading datasets successfully.")


    else: # Load the full dataset into memory
        print("Loading the full dataset into memory.")
        train_trajectories = []
        with tqdm.tqdm(total=num_train_dataset_episodes, desc="Processing Training Episodes") as pbar:
            for train_episode in train_dataset:
                curr_train_trajectories, tapir_model = process_episode(
                    train_episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model
                )
                train_trajectories.extend(curr_train_trajectories)
                pbar.update(1)

        val_trajectories = []
        with tqdm.tqdm(total=num_val_dataset_episodes, desc="Processing Validation Episodes") as pbar:
            for val_episode in val_dataset:
                curr_val_trajectories, tapir_model = process_episode(
                    val_episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model
                )
                val_trajectories.extend(curr_val_trajectories)
                pbar.update(1)

        # Create a tf.data.Dataset from trajectories
        def unlazy_gen(data):
            for traj in data:
                yield traj
        train_tf_dataset = tf.data.Dataset.from_generator(
            lambda: unlazy_gen(train_trajectories),
            output_types=output_types,
            output_shapes=output_shapes
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_tf_dataset = tf.data.Dataset.from_generator(
            lambda: unlazy_gen(val_trajectories),
            output_types=output_types,
            output_shapes=output_shapes
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print("Loaded the full dataset into memory.")

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
    parser.add_argument('--lazy_loading', type=bool, default=True, 
                        help='Whether to use lazy loading for the dataset (default: True)')

    args = parser.parse_args()

    # Build the dataset with the lazy_loading argument
    train_dataset, val_dataset = build_dataset(
        args.bridge_data_path, args.tracks_dir, args.tapir_model_checkpoint_fp, lazy_loading=args.lazy_loading
    )

    # Iterate through the dataset and process each sample
    for train_sample in train_dataset.take(5):  # Limit to 5 samples for demonstration
        process_data_sample(train_sample)
        print("Processed a train_data sample.")
        
    # Iterate through the dataset and process each sample
    for val_sample in val_dataset.take(5):  # Limit to 5 samples for demonstration
        process_data_sample(val_sample)
        print("Processed a val_data sample.")