# MyGen2Act
Just a quick and naive attempt at reproducing the method from the Gen2Act paper (not official)

# Installation

### First, install CUDA

First setup CUDA, I install it from the official wesbite [here](https://developer.nvidia.com/cuda-downloads). My exact personal setup is in the scratch section below. But the default should be

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### Setup conda env


Follow these steps to create and set up a Conda environment for this project.

```bash
conda create -n gen2act python=3.10 -y
conda activate gen2act
# git clone the repo
# cd to the repo root
pip install -r requirements.txt
```

# Usage

Just run the `video_processor.py` with (optional) path to your video 

NOTE: this will fail unless you have the saved tracks for the video already. 
```bash
cd path/to/MyGen2Act
conda activate gen2act
python video_processor.py --video_path /path/to/video.mp4
```

### Use random tracks (if you dont have preproccessed tracks)
This is super hacky but just uncomment this code here in the `video_processor.py` file. Basically take this part 
```python
# Uncomment this block to use random tracks instead of loading from files
# ----------------------------------------------------------
# num_frames = resized_video.shape[0]
# num_points = 1024  # Number of points to track
# tracks = np.random.rand(num_points, num_frames, 2) * np.array([og_video_width, og_video_height])
# visibles = np.random.rand(num_points, num_frames) > 0.5
# print("Using random tracks and visibles.")
# ----------------------------------------------------------


# Load preproccessed point tracking results  
tracking_results = load_tracking_results(video_path)
tracks = tracking_results['tracks']
visibles = tracking_results['visibles']
print(f"Tracks shape: {tracks.shape}")
print(f"Visibles shape: {visibles.shape}")
```
and change the comments to turn it into this 
```python
# Uncomment this block to use random tracks instead of loading from files
# ----------------------------------------------------------
num_frames = resized_video.shape[0]
num_points = 1024  # Number of points to track
tracks = np.random.rand(num_points, num_frames, 2) * np.array([og_video_width, og_video_height])
visibles = np.random.rand(num_points, num_frames) > 0.5
print("Using random tracks and visibles.")
# ----------------------------------------------------------


# Load preproccessed point tracking results  
# tracking_results = load_tracking_results(video_path)
# tracks = tracking_results['tracks']
# visibles = tracking_results['visibles']
print(f"Tracks shape: {tracks.shape}")
print(f"Visibles shape: {visibles.shape}")
```
and now it'll just use random numbers for the tracked points so you can run the code.

### Sample output

This is what the ouput looks like when I run the code

```bash
(gen2act) (base) ubuntu@b74b6fe5af4b:~/MyGen2Act$ python video_processor.py 
2024-10-21 13:21:16.166955: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-10-21 13:21:16.178152: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-10-21 13:21:16.181557: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-10-21 13:21:16.743279: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Video frames shape: (num_frames, height, width, channels) = (60, 480, 640, 3)
Resized video frames shape: (num_frames, height, width, channels) = (60, 224, 224, 3)
/opt/anaconda3/envs/gen2act/lib/python3.10/site-packages/transformers/models/vit/feature_extraction_vit.py:28: FutureWarning: The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.
  warnings.warn(
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['vit.pooler.dense.bias', 'vit.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.
Extracted features shape: (num_frames_processed, num_tokens, hidden_dim) =  (16, 197, 768)
i0_g shape: (1, num_tokens, hidden_dim) =  torch.Size([1, 197, 768])
PerceiverResampler output shape: (1, num_latents, hidden_dim) =  torch.Size([1, 64, 768])
zg shape: (1, num_latents, hidden_dim) =  torch.Size([1, 64, 768])
Tracks loaded from: /home/kasm-user/SimplerEnv-OpenVLA/octo_policy_video2_tracks.npz
Visibles loaded from: /home/kasm-user/SimplerEnv-OpenVLA/octo_policy_video2_visibles.npz
Tracks shape: (1024, 60, 2)
Visibles shape: (1024, 60)
P0_normalized shape: (1, num_points, 2) =  torch.Size([1, 1024, 2])
Ground truth tracks shape: (num_points, num_frames_processed, 2) =  (1024, 16, 2)
/opt/anaconda3/envs/gen2act/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
Auxiliary loss: 0.6787735223770142
(gen2act) (base) ubuntu@b74b6fe5af4b:~/MyGen2Act$ 
```

# Scratch space below (ignore)
ignore all below, these are details for my personal setup

so on kasm I go to this link for the CUDA install and get CUDA 12.6 

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network

and I run these commands as it says on the wesbite 

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

and then I also always get this error
```bash
Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
```

which I fix by doing this (only works after you do the cuda toolkit install above because of the keyring):
```bash
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev
```


just scratch below, ignore 
```
# check CUDA to be compatible with yours 
conda install -c conda-forge cudatoolkit-dev
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge cudnn=8

```