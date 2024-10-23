# data_loader.py

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from src.utils import construct_episode_label
from src.utils import load_tracking_results


def build_dataset(bridge_data_path, trajectory_length=8, next_actions_length=4, split='train[:10]', batch_size=1):

    """
    Builds a tf.data.Dataset of trajectories from the bridge dataset.

    Args:
        bridge_data_path (str): Path to the bridge dataset directory.
        trajectory_length (int): Length of the trajectory (number of steps).
        next_actions_length (int): Number of future actions to collect as labels.
        split (str): Dataset split to use (default: 'train[:10]').

    Returns:
        tf.data.Dataset: A dataset of trajectories.
    """
    print ("Building dataset")
    # Load the dataset
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    dataset = dataset_builder.as_dataset(split=split)

    trajectories = []

    # Loop over each episode in the dataset
    for episode_idx, episode in enumerate(dataset):
        # Collect the steps into a list
        steps = list(episode['steps'].as_numpy_iterator())
        num_steps = len(steps)

        # Sample 16 frames for the pretend gen video
        # Ensure we have at least 16 frames
        if num_steps >= 16:
            # Sample indices, ensuring the first and last frames are included
            episode_sampled_indices = [0]  # Start with the first frame
            if num_steps > 2:
                middle_indices = np.linspace(1, num_steps - 2, 14, dtype=int)
                episode_sampled_indices.extend(middle_indices.tolist())
            episode_sampled_indices.append(num_steps - 1)  # Include the last frame
        else:
            # If less than 16 frames, sample all frames
            episode_sampled_indices = list(range(num_steps))

        # Collect images of the sampled frames
        whole_episode_images = np.stack(
            [steps[idx]['observation']['image_0'] for idx in episode_sampled_indices], axis=0
        )
        
        episode_label = construct_episode_label(episode)

        # Generate trajectories of specified length, including all possible ones up to the last step
        for i in range(num_steps - trajectory_length + 1):
            # Collect the current trajectory of specified steps
            trajectory_steps = steps[i:i + trajectory_length]

            # Collect the next actions steps (or fewer if not available)
            next_actions_steps = steps[i + trajectory_length:i + trajectory_length + next_actions_length]

            # Collect images of the last trajectory_length observations
            trajectory_images = np.stack(
                [step['observation']['image_0'] for step in trajectory_steps], axis=0
            )

            # Collect actions for the trajectory steps
            trajectory_actions = np.stack(
                [step['action'] for step in trajectory_steps], axis=0
            )

            # Collect the next_actions_length actions and pad with zeros if not enough steps remain
            next_actions = np.zeros((next_actions_length, trajectory_actions.shape[1]), dtype=np.float32)
            for j, step in enumerate(next_actions_steps):
                next_actions[j] = step['action']

            # Collect discount
            trajectory_discount = np.stack(
                [step['discount'] for step in trajectory_steps], axis=0
            )

            # Collect is_first
            trajectory_is_first = np.stack(
                [step['is_first'] for step in trajectory_steps], axis=0
            )

            # Collect is_last
            trajectory_is_last = np.stack(
                [step['is_last'] for step in trajectory_steps], axis=0
            )

            # Collect is_terminal
            trajectory_is_terminal = np.stack(
                [step['is_terminal'] for step in trajectory_steps], axis=0
            )

            # Collect language_instruction
            language_instruction = trajectory_steps[0]['language_instruction']  # same for the whole trajectory

            # Collect reward
            trajectory_reward = np.stack(
                [step['reward'] for step in trajectory_steps], axis=0
            )
            
            # Collect preprocessed tracking data
            tracks_path = f"{bridge_data_path}/{episode_label}_tracks.npz"
            visibles_path = f"{bridge_data_path}/{episode_label}_visibles.npz"
            
            tracks, visibles = load_tracking_results(tracks_path, visibles_path)
            
            # sample the last 8 frames of the tracks and visibles
            trajectory_tracks = tracks[ : , i:i + trajectory_length, : ]
            trajectory_visibles = visibles[ : , i:i + trajectory_length]
            
            # sample the whole episode tracks and visibles
            whole_episode_tracks = tracks[ : , episode_sampled_indices, : ]
            whole_episode_visibles = visibles[ : , episode_sampled_indices ]
            
            
            

            # Create trajectory data
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
                # Include other necessary data if needed
            }

            # Add trajectory to the list
            trajectories.append(trajectory)

    # Now, create a tf.data.Dataset from trajectories
    def gen():
        for traj in trajectories:
            yield traj

    # Define the output types and shapes based on the first trajectory
    sample_trajectory = trajectories[0]
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

    output_shapes = {
        'trajectory_images': sample_trajectory['trajectory_images'].shape,
        'trajectory_actions': sample_trajectory['trajectory_actions'].shape,
        'next_actions': sample_trajectory['next_actions'].shape,
        'trajectory_discount': sample_trajectory['trajectory_discount'].shape,
        'trajectory_is_first': sample_trajectory['trajectory_is_first'].shape,
        'trajectory_is_last': sample_trajectory['trajectory_is_last'].shape,
        'trajectory_is_terminal': sample_trajectory['trajectory_is_terminal'].shape,
        'language_instruction': tf.TensorShape([]),  # scalar string
        'trajectory_reward': sample_trajectory['trajectory_reward'].shape,
        'whole_episode_images': sample_trajectory['whole_episode_images'].shape,
        'trajectory_tracks': sample_trajectory['trajectory_tracks'].shape,
        'trajectory_visibles': sample_trajectory['trajectory_visibles'].shape,
        'whole_episode_tracks': sample_trajectory['whole_episode_tracks'].shape,
        'whole_episode_visibles': sample_trajectory['whole_episode_visibles'].shape
    }

    trajectory_dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_types, output_shapes=output_shapes
    ).batch(batch_size)

    print("Dataset built successfully.")
    return trajectory_dataset


def process_data_sample(data_sample):
    """
    Dummy processing function for a dataset sample.
    """
    pass

if __name__ == '__main__':
    # Define the dataset path
    bridge_data_path = "/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/"

    # Build the dataset
    dataset = build_dataset(bridge_data_path)

    # Iterate through the dataset and process each sample
    for sample in dataset.take(5):  # Limit to 5 samples for demonstration
        process_data_sample(sample)
        print("Processed a data sample.")