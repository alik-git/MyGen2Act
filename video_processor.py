import torch
import torch.nn.functional as F
import torchvision.transforms as T
import mediapy as media
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel
import argparse

from tapnet.torch import tapir_model
from tapnet.utils import transforms, viz_utils

from pathlib import Path


# Define the ViT encoder (using ViT-B/16)
class VideoFeatureExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda:0'):
        self.device = device
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_frame(self, frame):
        # Ensure frame values are in [0, 1] before feeding to the feature extractor
        frame = (frame - frame.min()) / (frame.max() - frame.min())

        # Apply the same preprocessing as the ViT feature extractor
        inputs = self.feature_extractor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        return pixel_values

    def extract_features(self, video_frames):
        # Extract features from the list of video frames
        features = []
        for frame in video_frames:
            with torch.no_grad():
                pixel_values = self.preprocess_frame(frame)
                outputs = self.model(pixel_values)
                features.append(outputs.last_hidden_state.squeeze(0).cpu().numpy())

        # Stack all frame features into a single array
        return np.stack(features)


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


def main(video_path, resize_height=224, resize_width=224, num_points=5, device='cuda:0'):
    
    # Get original video width and height 
    og_video = media.read_video(video_path)
    og_video_height, og_video_width = og_video.shape[1:3]

    # Load and preprocess video frames
    resized_video, resized_video_frames = preprocess_video(og_video, resize_height, resize_width)

    # Initialize the ViT feature extractor with the specified device
    feature_extractor = VideoFeatureExtractor(device=device)

    # Extract features from video frames
    features = feature_extractor.extract_features(resized_video_frames)
    print("Extracted features shape:", features.shape)  # Shape: [num_frames, num_tokens, hidden_dim]
    
    tracking_results = load_tracking_results(video_path)
    tracks = tracking_results['tracks']
    visibles = tracking_results['visibles']

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
