import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tqdm

from src.utils import construct_episode_label, load_tracking_results
from src.pt_tracker import generate_tracking_files, load_tapir_model

import tensorflow_datasets as tfds
import tensorflow as tf

# Enable dynamic memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class TrajectoryDataset(Dataset):
    def __init__(self, processed_data_fp, device='cuda'):
        """
        PyTorch-compatible dataset for handling saved trajectories with lazy loading.

        Args:
            processed_data_fp (str): Directory for loading processed dataset.
            device (str): Device to load tensors onto (default: 'cuda').
        """
        self.processed_data_fp = processed_data_fp
        self.trajectories_fp = Path(processed_data_fp) / 'trajectories'
        self.whole_episode_images_fp = Path(processed_data_fp) / 'whole_episode_images'

        # Load paths to trajectory files
        self.file_paths = sorted(self.trajectories_fp.glob("*.npz"))
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        Loads a trajectory and corresponding images from the respective files on disk.
        """
        # Load trajectory data
        traj_file_path = self.file_paths[index]
        traj_data = np.load(traj_file_path)
        episode_idx = traj_data['episode_idx'].item() 

        # Load corresponding episode images
        saved_whole_episode_images_path = self.whole_episode_images_fp / f"whole_episode_images_{episode_idx}.npz"
        episode_images_data = np.load(saved_whole_episode_images_path)

        robot_traj_indices = traj_data['robot_traj_indices']  # Extract robot trajectory indices
        episode_sampled_indices = traj_data['episode_sampled_indices']  # Extract sampled episode indices
        episode_sampled_images = episode_images_data['images'][episode_sampled_indices]
        robot_trajectory_images = episode_images_data['images'][robot_traj_indices]

        # Prepare PyTorch tensors
        traj_tensors = {
            'trajectory_images': torch.tensor(robot_trajectory_images, device=self.device),
            'trajectory_actions': torch.tensor(traj_data['trajectory_actions'], device=self.device),
            'next_actions': torch.tensor(traj_data['next_actions'], device=self.device),
            'trajectory_discount': torch.tensor(traj_data['trajectory_discount'], device=self.device),
            'trajectory_is_first': torch.tensor(traj_data['trajectory_is_first'], device=self.device),
            'trajectory_is_last': torch.tensor(traj_data['trajectory_is_last'], device=self.device),
            'trajectory_is_terminal': torch.tensor(traj_data['trajectory_is_terminal'], device=self.device),
            'language_instruction': traj_data['language_instruction'].tobytes().decode('utf-8'),
            'trajectory_reward': torch.tensor(traj_data['trajectory_reward'], device=self.device),
            'episode_sampled_images': torch.tensor(episode_sampled_images, device=self.device),
            'trajectory_tracks': torch.tensor(traj_data['trajectory_tracks'], device=self.device),
            'trajectory_visibles': torch.tensor(traj_data['trajectory_visibles'], device=self.device),
            'whole_episode_tracks': torch.tensor(traj_data['whole_episode_tracks'], device=self.device),
            'whole_episode_visibles': torch.tensor(traj_data['whole_episode_visibles'], device=self.device),
            'episode_idx': episode_idx,
            'episode_label': traj_data['episode_label'].item()
        }

        return traj_tensors

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


def pytorch_process_episode(episode_idx, episode, trajectory_length, next_actions_length, tracks_dir, tapir_model_checkpoint_fp, tapir_model=None):
    """
    Process a single episode to extract trajectories and related data.
    """
    whole_episode_images = None
    trajectories = []
    steps = list(episode['steps'].as_numpy_iterator())
    num_steps = len(steps)

    # Skip episodes with fewer than 16 frames
    if num_steps < 16:
        print(f"Skipping episode due to insufficient frames: {num_steps}")
        return whole_episode_images, trajectories, tapir_model

    # Sample frames for the entire episode
    episode_sampled_indices = get_sample_indices(num_steps, num_samples=16)
    whole_episode_images = np.stack([step['observation']['image_0'] for step in steps], axis=0)
    whole_episode_images = whole_episode_images.astype(np.float32)

    episode_label = construct_episode_label(episode) # Just unique id 

    # Load tracks file 
    tracks_path = Path(tracks_dir) / f"{episode_label}_tracks.npz"
    visibles_path = Path(tracks_dir) / f"{episode_label}_visibles.npz"
    episode_tracks, episode_visibles = load_tracking_results(tracks_path, visibles_path)

    # If loading fails, try to regenerate the tracking files
    if episode_tracks is None or episode_visibles is None:
        print(f"Re-generating tracking files for episode: {episode_label}")
        if tapir_model is None:
            print("Initializing TAPIR model for missing tracking files...")
            tapir_model = load_tapir_model(tapir_model_checkpoint_fp)

        # Generate tracking files using the TAPIR model
        generate_tracking_files(episode, tapir_model, tracks_dir)

        # Attempt to re-load the tracking results
        episode_tracks, episode_visibles = load_tracking_results(tracks_path, visibles_path)

        if episode_tracks is None or episode_visibles is None:
            print(f"Failed to regenerate tracking files for episode: {episode_label}. Skipping this episode.")
            return trajectories, tapir_model

    # Process trajectories
    for i in range(num_steps - trajectory_length + 1):
        try:
            # Extract trajectory steps and next actions
            trajectory_steps = steps[i:i + trajectory_length]
            trajectory_indices = np.arange(i, i + trajectory_length)

            # get next 4 actions, including the current action, pad with zeros if necessary
            trajectory_actions = np.stack([step['action'] for step in trajectory_steps], axis=0)
            next_actions_steps = steps[i + trajectory_length - 1 :i + trajectory_length + next_actions_length - 1]
            next_actions = np.zeros((next_actions_length, trajectory_actions.shape[1]), dtype=np.float32)
            for j, step in enumerate(next_actions_steps):
                next_actions[j] = step['action']

            # get rest of the data sample 
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

            trajectory_tracks = episode_tracks[:, i:i + trajectory_length, :]
            trajectory_visibles = episode_visibles[:, i:i + trajectory_length]
            whole_episode_tracks = episode_tracks[:, episode_sampled_indices, :]
            whole_episode_visibles = episode_visibles[:, episode_sampled_indices]

            trajectory = {
                'episode_sampled_indices': episode_sampled_indices,
                'robot_traj_indices': trajectory_indices,
                'trajectory_actions': trajectory_actions,
                'next_actions': next_actions,
                'trajectory_discount': trajectory_discount,
                'trajectory_is_first': trajectory_is_first,
                'trajectory_is_last': trajectory_is_last,
                'trajectory_is_terminal': trajectory_is_terminal,
                'language_instruction': language_instruction,
                'trajectory_reward': trajectory_reward,
                'trajectory_tracks': trajectory_tracks,
                'trajectory_visibles': trajectory_visibles,
                'whole_episode_tracks': whole_episode_tracks,
                'whole_episode_visibles': whole_episode_visibles,
                'episode_idx' : episode_idx,  # Store episode index
                'episode_label': episode_label  # Store episode ID
            }
            trajectories.append(trajectory)
            
            # Ensure that all sampled indices are within the bounds of whole_episode_images
            assert max(episode_sampled_indices) < whole_episode_images.shape[0], (
                f"Error: sampled index out of bounds for episode {episode_idx}. "
                f"Max index: {max(episode_sampled_indices)}, Num images: {whole_episode_images.shape[0]}"
            )
        
        except Exception as e:
            print(f"Error processing trajectory {i} in episode {episode_label}: {e}. Skipping this trajectory.")
            continue
        
    return whole_episode_images, trajectories, tapir_model


def process_and_save_episodes(bridge_data_path, trajectory_length, next_actions_length, 
                              tracks_dir, tapir_model_checkpoint_fp, processed_data_fp, num_episodes, split='train'):
    """
    Processes episodes to extract trajectories and saves them to disk.
    """
    # Load episodes from the dataset builder
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    episodes = dataset_builder.as_dataset(split=split) # .take(num_episodes)
    
    # print number of episodes
    print(f"Number of episodes: {len(episodes)}")
    
    dataset_dir = Path(processed_data_fp) / f"saved_{split}_dataset_{num_episodes}_episodes"
    trajectories_fp = dataset_dir / 'trajectories'
    whole_episode_images_fp = dataset_dir / 'whole_episode_images'

    trajectories_fp.mkdir(parents=True, exist_ok=True)
    whole_episode_images_fp.mkdir(parents=True, exist_ok=True)

    tapir_model = None

    with tqdm.tqdm(total=len(episodes), desc="Processing Episodes") as pbar:
        for episode_idx, episode in enumerate(episodes):

            whole_episode_images, trajectories, tapir_model = pytorch_process_episode(
                episode_idx, episode, trajectory_length, next_actions_length, 
                tracks_dir, tapir_model_checkpoint_fp, tapir_model
            )

            if len(trajectories) == 0:
                pbar.update(1)
                continue
            
            # Save episode images
            whole_episode_images_path = whole_episode_images_fp / f"whole_episode_images_{episode_idx}.npz"
            np.savez_compressed(whole_episode_images_path, images=whole_episode_images, episode_idx=episode_idx)

            # Save each trajectory as a separate file
            for traj_idx, trajectory in enumerate(trajectories):
                traj_file_path = trajectories_fp / f"episode_{episode_idx}_trajectory_{traj_idx}.npz"
                np.savez_compressed(traj_file_path, **trajectory)
                # print(f"Saved trajectory to: {traj_file_path}")

            pbar.update(1)


def build_dataloader_from_fp(processed_data_fp, split='train', batch_size=32, num_workers=0, shuffle=True, num_episodes=5):
    """
    Builds a PyTorch DataLoader from a processed dataset.

    Args:
        processed_data_fp (str): Directory for loading processed dataset.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the dataset when loading.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset_dir = Path(processed_data_fp) / f"saved_{split}_dataset_{num_episodes}_episodes"
    dataset = TrajectoryDataset(dataset_dir)
    if len(dataset) == 0:
        raise ValueError(f"No valid files found in directory: {dataset_dir}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build PyTorch dataset from episodes.')
    parser.add_argument('--bridge_data_path', type=str, required=True, help='Path to bridge dataset directory.')
    parser.add_argument('--tracks_dir', type=str, required=True, help='Directory for point tracking data.')
    parser.add_argument('--tapir_model_checkpoint_fp', type=str, required=True, help='Path to TAPIR model checkpoint.')
    parser.add_argument('--processed_data_fp', type=str, default="/tmp/pytorch_dataset", help='Directory for saving/loading processed dataset.')
    parser.add_argument('--num_episodes', type=int, required=True, help='Number of episodes to process.')
    parser.add_argument('--save_processed_data', action='store_true', help='Whether to save the processed dataset.')
    parser.add_argument('--load_processed_data', action='store_true', help='Whether to load the processed dataset from disk.')
    parser.add_argument('--trajectory_length', type=int, default=8, help='Number of frames in each trajectory.')
    parser.add_argument('--next_actions_length', type=int, default=4, help='Number of future actions to collect as labels.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset when loading.')
    parser.add_argument('--split', type=str, required=True, help='Dataset split to use for processing.')

    args = parser.parse_args()
    
    # print args and their value 
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # If processing and saving a new dataset
    if args.save_processed_data and not args.load_processed_data:
        print(f"Processing and saving dataset to {args.processed_data_fp}...")
        process_and_save_episodes(
            bridge_data_path=args.bridge_data_path,
            trajectory_length=args.trajectory_length, 
            next_actions_length=args.next_actions_length, 
            tracks_dir=args.tracks_dir, 
            tapir_model_checkpoint_fp=args.tapir_model_checkpoint_fp, 
            processed_data_fp=args.processed_data_fp, 
            num_episodes=args.num_episodes, 
            split=args.split
        )

    print("Loading saved dataset...")
    dataloader = build_dataloader_from_fp(
        args.processed_data_fp, split=args.split, batch_size=args.batch_size, shuffle=args.shuffle, num_episodes=args.num_episodes
    )

    # Iterate through the DataLoader for demonstration
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch size: {len(batch)}")
        if batch_idx >= 5:
            break