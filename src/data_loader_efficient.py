# data_loader.py

import tensorflow_datasets as tfds
import tensorflow as tf

def build_dataset(bridge_data_path, trajectory_length=8, next_actions_length=4, split='train[:10]'):
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
    # Load the dataset
    dataset_builder = tfds.builder_from_directory(bridge_data_path)
    episodes_dataset = dataset_builder.as_dataset(split=split)

    # Function to process each episode
    def process_episode(episode):
        steps_dataset = episode['steps']

        # Map steps to extract needed data
        def map_step(step):
            return {
                'image_0': step['observation']['image_0'],
                'action': step['action'],
                'discount': step['discount'],
                'is_first': step['is_first'],
                'is_last': step['is_last'],
                'is_terminal': step['is_terminal'],
                'language_instruction': step['language_instruction'],
                'reward': step['reward']
            }

        steps_dataset = steps_dataset.map(map_step, num_parallel_calls=tf.data.AUTOTUNE)

        # Collect images of the whole episode
        # We can collect images using steps_dataset.map and batch
        whole_episode_images = steps_dataset.map(lambda x: x['image_0'])
        whole_episode_images = whole_episode_images.batch(1000000)  # Assuming episodes are shorter than 1,000,000 steps
        whole_episode_images = whole_episode_images.take(1)  # Take the first (and only) batch

        # Cache the whole episode images to avoid recomputing
        whole_episode_images = whole_episode_images.cache()

        # Function to generate trajectories within an episode
        def generate_trajectories():
            # We need to collect steps into a buffer to create sequences
            steps_buffer = []

            for step in steps_dataset.as_numpy_iterator():
                steps_buffer.append(step)

            num_steps = len(steps_buffer)

            # Convert whole_episode_images to a numpy array
            whole_images = tf.data.experimental.get_single_element(whole_episode_images)
            whole_images = whole_images.numpy()

            for i in range(num_steps - trajectory_length + 1):
                trajectory_steps = steps_buffer[i:i + trajectory_length]

                # Collect the next actions steps (or fewer if not available)
                next_actions_steps = steps_buffer[i + trajectory_length:i + trajectory_length + next_actions_length]

                # Collect images of the last trajectory_length observations
                trajectory_images = [step['image_0'] for step in trajectory_steps]
                trajectory_images = tf.stack(trajectory_images)

                # Collect actions for the trajectory steps
                trajectory_actions = [step['action'] for step in trajectory_steps]
                trajectory_actions = tf.stack(trajectory_actions)

                # Collect the next_actions_length actions and pad with zeros if not enough steps remain
                next_actions_list = [step['action'] for step in next_actions_steps]
                # Pad if necessary
                if len(next_actions_list) < next_actions_length:
                    pad_length = next_actions_length - len(next_actions_list)
                    padding = tf.zeros([pad_length, trajectory_actions.shape[1]], dtype=trajectory_actions.dtype)
                    next_actions_list.extend([padding.numpy()] * pad_length)
                next_actions = tf.stack(next_actions_list)

                # Collect other trajectory data
                trajectory_discount = tf.stack([step['discount'] for step in trajectory_steps])
                trajectory_is_first = tf.stack([step['is_first'] for step in trajectory_steps])
                trajectory_is_last = tf.stack([step['is_last'] for step in trajectory_steps])
                trajectory_is_terminal = tf.stack([step['is_terminal'] for step in trajectory_steps])
                trajectory_reward = tf.stack([step['reward'] for step in trajectory_steps])
                language_instruction = trajectory_steps[0]['language_instruction']

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
                    'whole_episode_images': whole_images,
                }

                yield trajectory

        return tf.data.Dataset.from_generator(
            generate_trajectories,
            output_types={
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
            },
            output_shapes={
                'trajectory_images': tf.TensorShape([trajectory_length, None, None, None]),
                'trajectory_actions': tf.TensorShape([trajectory_length, None]),
                'next_actions': tf.TensorShape([next_actions_length, None]),
                'trajectory_discount': tf.TensorShape([trajectory_length]),
                'trajectory_is_first': tf.TensorShape([trajectory_length]),
                'trajectory_is_last': tf.TensorShape([trajectory_length]),
                'trajectory_is_terminal': tf.TensorShape([trajectory_length]),
                'language_instruction': tf.TensorShape([]),
                'trajectory_reward': tf.TensorShape([trajectory_length]),
                'whole_episode_images': tf.TensorShape([None, None, None, None]),
            }
        )

    # Apply process_episode to each episode in the dataset
    trajectories_dataset = episodes_dataset.flat_map(process_episode)

    return trajectories_dataset.prefetch(tf.data.AUTOTUNE)
