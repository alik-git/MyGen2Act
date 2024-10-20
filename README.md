# MyGen2Act
Just a quick and naive attempt at reproducing the method from the Gen2Act paper (not official)

# Installation

Follow these steps to create and set up a Conda environment for this project.

```
conda create -n gen2act python=3.10 -y
conda activate gen2act
# cd to the repo root
pip install -r requirements.txt
```


### Optional
and then I also had to install cuda from the official site here https://developer.nvidia.com/cuda-downloads

and maybe you might also need 
```bash
sudo apt-get install libcudnn8 libcudnn8-dev
```

### Scratch (ignore)
ignore all below 

```
# check CUDA to be compatible with yours 
conda install -c conda-forge cudatoolkit-dev
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge cudnn=8

```