# From-Rigging-to-Waving
This is the official PyTorch implementation of the 2025 paper: From Rigging to Waving: 3D-Guided Diffusion for Natural Animation of Hand-Drawn Characters

## Installation

### **1. Install Python Dependencies**

### **2. Download the pretrained checkpoints**

To download the UniAnimate models, please follow the commands provided in the [UniAnimate](https://github.com/ali-vilab/UniAnimate). After that, you can download our domain-adapted model from [Baidu](https://pan.baidu.com/s/14GqXTFgK4d8i5wCVOwWtBA).（pwd: r5do）
 

Once downloaded, move the checkpoints to the `checkpoints/` directory. The model weights will be organized in the `./checkpoints/` directory as follows:

```./checkpoints/
|---- open_clip_pytorch_model.bin
|---- unianimate_16f_32f_non_ema_223000.pth 
|---- v2-1_512-ema-pruned.ckpt
└---- rigging2waving_non_ema_00040000.pth
```

## Inference

### **1. Run the Model to Generate Videos**

To generate video clips (32 frames), execute the following command:

```bash
python inference.py --cfg configs/infer.yaml
```

## Training

### **1. Prepare Datasets
All training dataset can be download from [Baidu](https://pan.baidu.com/s/14GqXTFgK4d8i5wCVOwWtBA).（pwd: r5do）
After downloading, extract the files and place them in the data folder:

```./data/
└---- rigging2waving_dataset_train
    |-- 0a4ff03c912a4e5487e74e05423f3c6d/  # A hand-drawn character
    |   |-- blender_render/  # Animation sequance
    |   └---char/ # Reference
```

### **2. Run Training Scripts
To train the domain-adapted model for hand-drawn characters, use the following command:

```bash
python train.py --cfg configs/train.yaml
```


## TODO List

- [ ] Add  long video generation.
