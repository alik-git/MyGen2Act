"""This files sole purpose is to check that the predicted point tracks make sense, it will save mp4 files with visualizations of the tracks"""
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from pathlib import Path
import mediapy as media
from tapnet.utils import viz_utils

from utils import construct_episode_label

# Your dataset path
bridge_data_path = "/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/"
output_folder = "/home/kasm-user/Downloads/bridge_dataset_tracks_viz"
output_folder = Path(output_folder)
output_folder.mkdir(parents=True, exist_ok=True)

# Load the dataset
dataset_builder = tfds.builder_from_directory(bridge_data_path)
dataset = dataset_builder.as_dataset(split='train[0:10]')

for episode in dataset:
    # Get images from the episode
    images = [step['observation']['image_0'] for step in episode['steps']]
    images = [Image.fromarray(image.numpy()) for image in images]
    # print(f"Image resolution: {images[0].size}")
    # print(f"Number of images: {len(images)}")
    
    # Convert images to frames (NumPy array)
    video = np.stack([np.array(image) for image in images], axis=0)  # [num_frames, height, width, 3]
    num_frames, height, width, _ = video.shape

    episode_label = construct_episode_label(episode)
    
    
    tracks_output_fp = f"/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/{episode_label}_tracks.npz"
    visibles_output_fp = f"/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/{episode_label}_visibles.npz"
    
    # Load the tracks and visibles
    if not Path(tracks_output_fp).is_file() or not Path(visibles_output_fp).is_file():
        print(f"Tracks or visibles not found for episode {episode_label}, skipping.")
        continue

    tracks_data = np.load(tracks_output_fp)
    tracks = tracks_data['tracks']  # [num_points, num_frames, 2]
    
    visibles_data = np.load(visibles_output_fp)
    visibles = visibles_data['visibles']  # [num_points, num_frames]
    
    # Visualize the tracks
    # Ensure that tracks are in the correct shape and coordinates
    # tracks: [num_points, num_frames, 2], coordinates in pixel space
    
    # Paint the point tracks
    video_viz = viz_utils.paint_point_track(video, tracks, visibles)
    
    # Save the visualization as an MP4 file
    video_output_fp = f"{output_folder}/{episode_label}_tracks_viz.mp4"
    media.write_video(video_output_fp, video_viz, fps=10)
    print(f"Saved video with tracks to {video_output_fp}\n")
