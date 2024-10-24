"""
python full_pipeline.py --bridge_data_path /nfs/scratch/pawel/octo/octo/bridge_dataset/1.0.0/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import argparse


from tqdm import tqdm


# Import build_dataset from data_loader.py
from src.action_prediction_transformer import ActionPredictionTransformer
from src.data_loader import build_dataset


from src.vit_ft_extractor import VideoFeatureExtractor
from src.perceiver_resampler import PerceiverResampler
from src.track_pred_transformer import TrackPredictionTransformer


from src.utils import track_memory


from src.utils import save_checkpoint, load_checkpoint

def run_epoch(epoch_num, dataloader, models, action_criterion, optimizer, device, track_memory_flag, training=True):
    """
    Run a single epoch for training or validation.

    Args:
        dataloader: DataLoader for the dataset.
        models (dict): Dictionary of models to use.
        action_criterion: Loss function for action prediction.
        optimizer: Optimizer for model updates.
        device (str): Device to run on.
        track_memory_flag (bool): Whether to track memory usage.
        training (bool): If True, run training; else, run validation.

    Returns:
        tuple: Average total loss, action loss, auxiliary loss.
    """
    
    if training:
        mode = 'Train'
        for model in models.values():
            model.train()
    else:
        mode = 'Validation'
        for model in models.values():
            model.eval()
    
    feature_extractor = models['feature_extractor']
    gen_video_perceiver_resampler = models['gen_video_perceiver_resampler']
    robot_video_perceiver_resampler = models['robot_video_perceiver_resampler']
    gen_video_track_predictor = models['gen_video_track_predictor']
    robot_video_track_predictor = models['robot_video_track_predictor']
    action_predictor = models['action_predictor']
    
    epoch_total_loss, epoch_action_loss, epoch_aux_loss = 0, 0, 0

    # Get total number of batches
    print("counting batches")
    epoch_total_batches = sum(1 for _ in dataloader)
    print("done counting batches")
    
    
    # Disable gradient calculation in validation
    context = torch.no_grad() if not training else torch.enable_grad()


    # Progress bar with avg loss
    with tqdm(enumerate(dataloader), total=epoch_total_batches, desc=f"Epoch {epoch_num + 1}", leave=True) as pbar:
        for batch_idx, batch in pbar:




            # Get generated and robot videos
            generated_frames = batch['whole_episode_images'].numpy() # (batch_size, num_frames_gen, H, W, C)
            robot_frames = batch['trajectory_images'].numpy() # (batch_size, num_frames_robot, H, W, C)

            # Get ground-truth tracks
            tracks_whole_episode = batch['whole_episode_tracks'].numpy() # (batch_size, num_points, num_frames_gen, 2)
            tracks_robot = batch['trajectory_tracks'].numpy() # (batch_size, num_points, num_frames_robot, 2)

            # Ground truth actions
            gt_actions = batch['next_actions'].numpy()  # Shape: (batch_size, num_future_actions, action_dim)
            gt_actions = torch.from_numpy(gt_actions).to(device).float() # Shape: (batch_size, num_future_actions, action_dim)


            # --- ViT Extraction for Generated and Robot Videos ---
            # Extract features using ViT for both generated and robot videos
            gen_features = feature_extractor.forward(generated_frames)  # (batch_size, num_frames_gen, num_tokens_per_frame, hidden_dim)
            robot_features = feature_extractor.forward(robot_frames)  # (batch_size, num_frames_robot, num_tokens_per_frame, hidden_dim)

            # Initial ViT features from the first frame
            i0_g = gen_features[:, 0, :, :]  # (batch_size, num_tokens, hidden_dim), first frame features for generated video
            i0_r = robot_features[:, 0, :, :]  # (batch_size, num_tokens, hidden_dim), first frame features for robot video

            # Flatten features for Perceiver Resampler
            batch_size, num_frames_gen, num_tokens, hidden_dim = gen_features.shape
            gen_features_flat = gen_features.view(batch_size, -1, hidden_dim)  # (batch_size, num_frames_gen * num_tokens, hidden_dim)
            robot_features_flat = robot_features.view(batch_size, -1, hidden_dim)  # (batch_size, num_frames_robot * num_tokens, hidden_dim)


            # --- Perceiver Resampler Compression ---
            # Compress features for both generated and robot videos
            z_g = gen_video_perceiver_resampler(gen_features_flat)  # Shape: (batch_size, num_latents, hidden_dim), latent tokens for generated video
            z_r = robot_video_perceiver_resampler(robot_features_flat)  # Shape: (batch_size, num_latents, hidden_dim), latent tokens for robot video


            # --- Track Prediction for Generated and Robot Videos ---
            # Prepare initial points (P0) for track prediction
            P0_g = tracks_whole_episode[:, :, 0, :]  # Shape: (batch_size, num_points, 2), initial points for generated video
            P0_r = tracks_robot[:, :, 0, :]  # Shape: (batch_size, num_points, 2), initial points for robot video

            # Normalize coordinates to be between 0 and 1
            img_height, img_width = generated_frames.shape[2], generated_frames.shape[3]
            P0_g_normalized = P0_g / np.array([img_width, img_height])  # Shape: (batch_size, num_points, 2)
            P0_r_normalized = P0_r / np.array([img_width, img_height])  # Shape: (batch_size, num_points, 2)

            # Convert initial points to PyTorch tensors
            P0_g_normalized_tensor = torch.tensor(P0_g_normalized, device=device).float()  # Shape: (batch_size, num_points, 2)
            P0_r_normalized_tensor = torch.tensor(P0_r_normalized, device=device).float()  # Shape: (batch_size, num_points, 2)

            # Prepare ground-truth tracks as PyTorch tensors
            gt_tracks_g = tracks_whole_episode / np.array([img_width, img_height])  # Shape: (batch_size, num_points, num_frames_gen, 2)
            gt_tracks_r = tracks_robot / np.array([img_width, img_height])  # Shape: (batch_size, num_points, num_frames_robot, 2)
            gt_tracks_g_tensor = torch.tensor(gt_tracks_g, device=device).float()  # Shape: (batch_size, num_points, num_frames_gen, 2)
            gt_tracks_r_tensor = torch.tensor(gt_tracks_r, device=device).float()  # Shape: (batch_size, num_points, num_frames_robot, 2)

            # Predict tracks using TrackPredictionTransformer for both generated and robot videos
            predicted_tracks_g = gen_video_track_predictor(P0_g_normalized_tensor, i0_g, z_g)  # Shape: (batch_size, num_points, num_frames_gen, 2)
            predicted_tracks_r = robot_video_track_predictor(P0_r_normalized_tensor, i0_r, z_r)  # Shape: (batch_size, num_points, num_frames_robot, 2)


            # Compute auxiliary loss for track predictions
            aux_loss_g = F.mse_loss(predicted_tracks_g, gt_tracks_g_tensor)
            aux_loss_r = F.mse_loss(predicted_tracks_r, gt_tracks_r_tensor)
            batch_aux_loss = aux_loss_g + aux_loss_r

            # --- Action Prediction ---
            # Use action predictor with latent tokens from both videos
            action_logits = action_predictor(z_g, z_r)  # Shape: (batch_size, num_future_actions, action_dim, num_bins), predicted actions

            # Discretize actions
            # These normalization values can be found in the dataset statistics files in the dataset, for example
            # https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/action_proprio_stats_7d6a416829d818b733e7342f225f3c522a8265a5224e0175f2ab28e26a932ff1.json
            action_min = torch.tensor([-0.1791, -0.2553, -0.0681, -0.4487, -0.3450, -6.2708, 0.0], device=device)  # Shape: (action_dim,)
            action_max = torch.tensor([0.1117, 0.2004, 0.1396, 0.5934, 0.3536, 6.2598, 1.0], device=device)  # Shape: (action_dim,)
            gt_actions_normalized = (gt_actions - action_min) / (action_max - action_min)  # Shape: (batch_size, num_future_actions, action_dim)
            gt_actions_normalized = torch.clamp(gt_actions_normalized, 0.0, 1.0)
            gt_actions_discrete = (gt_actions_normalized * (action_predictor.num_bins - 1)).long() # 256 bins


            # Compute action prediction loss
            action_logits_flat = action_logits.view(-1, action_predictor.num_bins)  # Shape: (batch_size * num_future_actions * action_dim, num_bins)
            gt_actions_flat = gt_actions_discrete.view(-1)  # Shape: (batch_size * num_future_actions * action_dim)
            batch_action_loss = action_criterion(action_logits_flat, gt_actions_flat)  # Scalar, action prediction loss


            # --- Total Loss Computation ---
            # Without weights, after a few epochs:
            # batch_aux_loss averages around 0.005.
            # batch_action_loss averages around 3.0.
            # so chatGPT suggested these weights
            aux_loss_weight = 200.0
            action_loss_weight = 0.33
            batch_total_loss = (aux_loss_weight * batch_aux_loss) \
                + (action_loss_weight * batch_action_loss)


            # Backpropagation and optimization
            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()

            # Accumulate losses for tracking
            epoch_total_loss += batch_total_loss.item()
            epoch_action_loss += batch_action_loss.item()
            epoch_aux_loss += batch_aux_loss.item()


            # Calculate running average loss
            batch_running_avg_loss = epoch_total_loss / (batch_idx + 1)
            batch_running_avg_action_loss = epoch_action_loss / (batch_idx + 1)
            batch_running_avg_aux_loss = epoch_aux_loss / (batch_idx + 1)


            # Update TQDM progress bar
            pbar.set_postfix({
                'Avg Loss': f'{batch_running_avg_loss:.4f}',
                'Avg Action Loss': f'{batch_running_avg_action_loss:.4f}',
                'Avg Aux Loss': f'{batch_running_avg_aux_loss:.4f}'
            })
            k=1

    # Final average losses for the epoch
    avg_total_loss = epoch_total_loss / max(1, epoch_total_batches)
    avg_action_loss = epoch_action_loss / max(1, epoch_total_batches)
    avg_aux_loss = epoch_aux_loss / max(1, epoch_total_batches)

    print(f"{mode} - Avg Total Loss: {avg_total_loss:.4f}, Avg Action Loss: {avg_action_loss:.4f}, Avg Aux Loss: {avg_aux_loss:.4f}")
    return avg_total_loss, avg_action_loss, avg_aux_loss



def train_model(
   bridge_data_path, tracks_dir, tapir_model_checkpoint_fp, save_dir, device='cuda:0', batch_size=1, epochs=1000, patience=10, track_memory_flag=False, checkpoint_path=None
):
   """
   Main function to build the dataset and  train models.


   Args:
       bridge_data_path (str): Path to the bridge dataset directory.
       tracks_dir (str): Path to the point tracks directory.
       device (str): Device to use (e.g., cuda:0, cuda:1, or cpu).
       batch_size (int): Batch size for processing trajectories.
   """
   print("==== Starting Video Processing Pipeline ====")
   print(f"Bridge Data Path: {bridge_data_path}")
   print(f"Tracks Directory: {tracks_dir}")
   print(f"Device: {device}")
   print(f"Batch Size: {batch_size}")


   # Build the dataset
   train_trajectories, val_trajectories = build_dataset(
       bridge_data_path=bridge_data_path,
       tracks_dir=tracks_dir,
       tapir_model_checkpoint_fp=tapir_model_checkpoint_fp,
       trajectory_length=8,
       next_actions_length=4,
       train_split='train[:10]',
       val_split='val[:2]',
       batch_size=batch_size
   )

   # Initialize models with memory profiling
   track_memory("Before initializing models", device, track_flag=track_memory_flag)


   # Feature Extractor
   feature_extractor = VideoFeatureExtractor(device=device)
  
   # Freeze ViT parameters to prevent updates during training
   for param in feature_extractor.parameters():
       param.requires_grad = False


   # Perceiver Resampler
   gen_video_perceiver_resampler = PerceiverResampler(
       dim=768, num_latents=64, depth=2, heads=8, dim_head=64, ff_mult=4
   ).to(device)
  
   robot_video_perceiver_resampler = PerceiverResampler(
       dim=768, num_latents=64, depth=2, heads=8, dim_head=64, ff_mult=4
   ).to(device)


   # Track Prediction Transformer for generated video
   gen_video_track_predictor = TrackPredictionTransformer(
       point_dim=2, hidden_dim=768, num_layers=6, num_heads=8, num_frames=16
   ).to(device)


   # Track Prediction Transformer for robot video
   robot_video_track_predictor = TrackPredictionTransformer(
       point_dim=2, hidden_dim=768, num_layers=6, num_heads=8, num_frames=8
   ).to(device)


   # Action Prediction Transformer
   action_predictor = ActionPredictionTransformer(
       hidden_dim=768,
       num_encoder_layers=6,
       num_decoder_layers=6,
       num_heads=8,
       action_dim=7,  # As per your action dimensions
       num_future_actions=4,  # Predict next 4 actions
       num_bins=256,
       dropout=0.1
   ).to(device)
   track_memory("After initializing models", device, track_flag=track_memory_flag)
  
   models = {
       'feature_extractor': feature_extractor,
       'gen_video_perceiver_resampler': gen_video_perceiver_resampler,
       'robot_video_perceiver_resampler': robot_video_perceiver_resampler,
       'gen_video_track_predictor': gen_video_track_predictor,
       'robot_video_track_predictor': robot_video_track_predictor,
       'action_predictor': action_predictor
   }




   # Loss function for action prediction
   action_criterion = nn.CrossEntropyLoss()
  
   config = {
       'learning_rate': 1e-4,
   }
  
   run_logs = {
   }


   # Optimizer
   optimizer = torch.optim.Adam(
       list(gen_video_perceiver_resampler.parameters()) +
       list(robot_video_perceiver_resampler.parameters()) +
       list(gen_video_track_predictor.parameters()) +
       list(robot_video_track_predictor.parameters()) +
       list(action_predictor.parameters()),
       lr=config['learning_rate']
   )
  
#    checkpoint_path = "/home/kasm-user/MyGen2Act/saved_checkpoints/epoch_5"
  
   # Load from checkpoint if provided
   if checkpoint_path:
       print(f"Loading checkpoint from {checkpoint_path}...")
       config, run_logs = load_checkpoint(checkpoint_path, device, models, optimizer)


       # print(f"Checkpoint loaded. Starting with best loss: {best_avg_loss:.4f}")


  
   # Initialize best_loss
   best_avg_loss = float('inf')  # Initialize best_loss at the start
   
   # Initialize best_loss for early stopping
   best_val_loss = float('inf')
   patience_counter = 0
  
   for epoch in range(epochs):
       print(f"\nEpoch {epoch + 1}/{epochs}")
       
       
       # Training phase
       train_total_loss, train_action_loss, train_aux_loss = run_epoch(
           epoch, train_trajectories, models, action_criterion, optimizer, device, track_memory_flag, training=True
       )
       
       # Validation phase
       val_total_loss, val_action_loss, val_aux_loss = run_epoch(
           epoch, val_trajectories, models, action_criterion, optimizer, device, track_memory_flag, training=False
       )
       
       # Log average losses
       run_logs[f'epoch_{epoch + 1}'] = {
           'train_total_loss': train_total_loss,
           'train_action_loss': train_action_loss,
           'train_aux_loss': train_aux_loss,
           'val_total_loss': val_total_loss,
           'val_action_loss': val_action_loss,
           'val_aux_loss': val_aux_loss
       }
       
       print(f"Epoch {epoch + 1} - Train Loss: {train_total_loss:.4f}, Val Loss: {val_total_loss:.4f}")
       
       # Early stopping
       if val_total_loss < best_val_loss:
           best_val_loss = val_total_loss
           patience_counter = 0
           print(f"New best model at epoch {epoch + 1} with validation loss {best_val_loss:.4f}.")
           save_checkpoint(epoch + 1, models, optimizer, config, run_logs, save_dir)
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
               break

       # Optionally save a checkpoint at the end of every epoch
    #    save_checkpoint(epoch + 1, models, optimizer, config, run_logs, save_dir)


if __name__ == "__main__":
   # Parse command-line arguments
   parser = argparse.ArgumentParser(description='Run video processing with specified GPU and batch size.')
   parser.add_argument('--bridge_data_path', type=str, default="/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/",
                       help='Path to the bridge dataset directory.')
   parser.add_argument('--tracks_dir', type=str, default="/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/",
                       help='Path to the point tracks directory.')
   parser.add_argument('--tapir_model_checkpoint_fp', type=str, default="/home/kasm-user/tapnet/checkpoints/bootstapir_checkpoint_v2.pt",
                       help='Path to the TAPIR model checkpoint file.')
   # add sav_dir arg for where to save trained models
   parser.add_argument('--save_dir', type=str, default='/home/kasm-user/MyGen2Act/saved_checkpoints', help='path to save trained model checkpoints')
   parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use, e.g., cuda:0, cuda:1, or cpu (default: cuda:0)')
   parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing trajectories (default: 1)')
   parser.add_argument('--epochs', type=int, default=1000)
   parser.add_argument('--patience', type=int, default=100)
   parser.add_argument('--track_memory_flag', type=bool, default=False,
                       help='Track memory usage (default: False)')
   parser.add_argument('--checkpoint_path', type=str, default=None,
   help='Path to a saved checkpoint to load from (if any).')
   args = parser.parse_args()


   train_model(
       bridge_data_path=args.bridge_data_path,
       tracks_dir=args.tracks_dir,
       tapir_model_checkpoint_fp=args.tapir_model_checkpoint_fp,
       save_dir=args.save_dir,
       device=args.device,
       batch_size=args.batch_size,
       epochs=args.epochs,
       patience=args.patience,
       track_memory_flag=args.track_memory_flag,
       checkpoint_path=args.checkpoint_path
   )



