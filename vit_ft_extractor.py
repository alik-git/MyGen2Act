
import torch
import torchvision.transforms as T
from transformers import ViTFeatureExtractor, ViTModel
import mediapy as media
import numpy as np

# Define the ViT encoder (using ViT-B/16)
class VideoFeatureExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(device)
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

def load_and_preprocess_video(video_path, resize_height=224, resize_width=224):
    # Load video using mediapy
    video = media.read_video(video_path)
    video = media.resize_video(video, (resize_height, resize_width))

    # Convert to RGB format and normalize
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_frames = [transform(frame).permute(1, 2, 0).numpy() for frame in video]
    return video_frames

# Example usage
if __name__ == "__main__":
    video_path = '/home/kasm-user/Uploads/Hand_picks_up_the_can.mp4'  # Path to your video file
    video_frames = load_and_preprocess_video(video_path)
    print(f"frames len : {len(video_frames)}")

    # Initialize the ViT feature extractor
    feature_extractor = VideoFeatureExtractor()

    # Extract features from video frames
    features = feature_extractor.extract_features(video_frames)

    print("Extracted features shape:", features.shape)  # Shape: [num_frames, num_tokens, hidden_dim]
