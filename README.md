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

### Install the point tracking repo
We need the PyTorch version of the "BootsTAPIR" model, so download the checkpoint for that [here](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt).
```
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
```
For more details, follow the instructions on their readme [here](https://github.com/google-deepmind/tapnet).

### Setup conda env
Follow these steps to create and set up a Conda environment for this project.

```bash
conda create -n gen2act python=3.10 -y
conda activate gen2act
git clone https://github.com/alik-ai/MyGen2Act.git
cd MyGen2Act
pip install -r requirements.txt
```

### Download the bridge dataset

Follow the instructions for that from their [website](https://github.com/rail-berkeley/bridge_data_v2) 

# Usage

### First compute the point tracking
Use the `pt_tracker.py` file for this. Replace the paths to match yours 

```bash
python src/pt_tracker.py --bridge_data_path /nfs/scratch/pawel/octo/octo/bridge_dataset/1.0.0 \
    --model_checkpoint_fp checkpoints/bootstapir_checkpoint_v2.pt
``` 

This will save point tracking files in the `.npz` format in the same location as your bridge dataset.

### Full loop
Just run the `full_pipeline.py` with the path to the bridge dataset. So far I have stuff implmented all the way up until the auxiliary loss for the each of the videos. Also since I haven't yet figured out how to generate the video, I just make a copy of the robot video and pretend its a generated video.

NOTE: this will fail unless you have the saved tracks for the video already. 
```bash
cd path/to/MyGen2Act
conda activate gen2act
python video_processor.py --bridge_data_path /home/kasm-user/alik_local_data/bridge_dataset/1.0.0/
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