import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
from pathlib import Path

from utils import construct_episode_label

# Should look like "path/to/bridge_dataset/1.0.0/"
bridge_data_path = "/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/"

dataset_builder = tfds.builder_from_directory(bridge_data_path)
# print(dataset_builder.info.features['steps']['observation'])

dataset = dataset_builder.as_dataset(split='train[0:10]')

for episode in dataset:
    # Get images from the episode
    images = [step['observation']['image_0'] for step in episode['steps']]
    images = [Image.fromarray(image.numpy()) for image in images]
    print(f"Image resolution: {images[0].size}")
    print(f"Number of images: {len(images)}")
    
    episode_label = construct_episode_label(episode)
    
    saved_tracks_fp = f"/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/{episode_label}_tracks.npz"
    saved_visibles_fp = f"/home/kasm-user/alik_local_data/bridge_dataset/1.0.0/{episode_label}_visibles.npz"
    
    curr_step = next(iter(episode['steps']))
    # curr_step.keys()
    # dict_keys(['action', 'discount', 'is_first', 'is_last', 'is_terminal', 'language_embedding', 'language_instruction', 'observation', 'reward'])
    
    # curr_step['action']
    # <tf.Tensor: shape=(7,), dtype=float32, numpy=
    # array([-5.1145643e-11, -1.2105482e-10, -9.3151029e-11,  2.5257037e-07,
    #        -9.5844953e-08, -5.6845714e-08,  1.0000000e+00], dtype=float32)>
    
    # curr_step['language_instruction']
    # <tf.Tensor: shape=(), dtype=string, numpy=b'put small spoon from basket to tray'>
    
    curr_lang_instruction = curr_step['language_instruction'].numpy().decode('utf-8')
    # curr_lang_instruction
    # 'put small spoon from basket to tray'
    

    
    

    break
















episode = next(iter(dataset))
images = [step['observation']['image_0'] for step in episode['steps']]
images = [Image.fromarray(image.numpy()) for image in images]

# Define the path to save images
save_folder = Path("/home/kasm-user/alik_local_data/first_episode_images/")
save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

# Iterate through the episodes in the dataset
for i, episode in enumerate(dataset):
    # Get the first step in the episode
    curr_steps = next(iter(episode['steps']))

    # Extract the language instruction and decode it
    instruction = curr_steps['language_instruction'].numpy().decode('utf-8')

    # Replace spaces and special characters in the instruction to make it a valid filename
    instruction_clean = instruction.replace(" ", "_").replace("/", "-")
    
    if instruction_clean == "":
        instruction_clean = "no_instruction"

    # Get the first image from the episode
    first_image = curr_steps['observation']['image_0'].numpy()

    # Convert to PIL Image and save
    image = Image.fromarray(first_image)
    image_path = save_folder / f"{instruction_clean}.png"
    image.save(image_path)

    print(f"Saved image {i+1} as: {image_path}")

curr_steps = next(iter(episode['steps']))

for elem in next(iter(episode['steps'])).items():
    print(elem)
    
# Create an empty list to store language instructions
language_instructions = []

# Iterate through the episodes in the dataset
for episode in dataset:
    # Iterate through the steps within each episode
    for step in episode['steps']:
        # Extract the language instruction and decode it
        instruction = step['language_instruction'].numpy().decode('utf-8')
        language_instructions.append(instruction)
        break

# Print all the language instructions
print("All Language Instructions:")
for i, instruction in enumerate(language_instructions):
    print(f"{i+1}: {instruction}")

k=2

fig, ax = plt.subplots()

def update_frame(i):
    ax.imshow(images[i])
    ax.set_title(f"Frame {i+1}")
    ax.axis('off')

ani = animation.FuncAnimation(fig, update_frame, frames=len(images), interval=200)  # 200ms interval
plt.show()

# Display each image in a separate plot
for i, image in enumerate(images):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Image {i+1}")
    plt.axis('off')  # Hide axes
    plt.show()
    break



j=2