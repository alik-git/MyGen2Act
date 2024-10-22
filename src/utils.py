from pathlib import Path

def construct_episode_label(episode):
    episode_id = int(episode['episode_metadata']['episode_id'].numpy())
    episode_fp = episode['episode_metadata']['file_path'].numpy().decode('utf-8')
    episode_fp = str(Path(episode_fp).with_suffix(''))
    episode_fp_clean = episode_fp.replace("/", "_")
    episode_label = f"episode_{episode_id}___{episode_fp_clean}"
    
    return episode_label


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