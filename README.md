# From-Rigging-to-Waving
This is the official PyTorch implementation of the 2025 paper: From Rigging to Waving: 3D-Guided Diffusion for Natural Animation of Hand-Drawn Characters

## Installation

### **1. Install Python Dependencies**

### **2. Download the pretrained checkpoints**

To download the UniAnimate models, please follow the commands provided in the [UniAnimate GitHub repository](https://github.com/ali-vilab/UniAnimate). After that, you can download our domain-adapted model from **.

Once downloaded, move the checkpoints to the `checkpoints/` directory. The model weights will be organized in the `./checkpoints/` directory as follows:

```./checkpoints/
|---- open_clip_pytorch_model.bin
|---- unianimate_16f_32f_non_ema_223000.pth 
|---- v2-1_512-ema-pruned.ckpt
└---- rigging2waving_non_ema_00040000.pth

## Inference

### **1. Run the Model to Generate Videos**

To generate video clips (32 frames), execute the following command:

```bash
python inference.py --cfg configs/infer.yaml

## Training


```bash
python train.py --cfg configs/train.yaml


## TODO List

- [ ] Add  long video generation.
