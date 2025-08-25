import os
import cv2
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from io import BytesIO
import glob
from torch.utils.data import Dataset, DataLoader
from utils.registry_class import DATASETS
import torchvision.transforms as transforms
from .random_mask import create_random_shape_with_random_motion
from torchvision.transforms import InterpolationMode
from concurrent.futures import ThreadPoolExecutor



def RGBA2RGB(image, bg_color=(255, 255, 255)):
    background = Image.new("RGB", image.size, bg_color)
    background.paste(image, mask=image.split()[3])
    return background

@DATASETS.register_class()
class VideoDataset(Dataset):
    def __init__(self, 
            data_list, 
            vit_resolution=[224, 224],
            resolution=(512, 768),
            max_frames=1,
            sample_fps=4,
            transforms=None,
            transforms_pose=None,
            vit_transforms=None,
            kernel_size=15,
            iters=[8,20],
            weights=[1,0,2],
            turn_background_mask=False,
            **kwargs):
        if not isinstance(data_list, str):
            raise ValueError("data_list should be a string representing the path to the dataset")
        
        self.max_frames = max_frames
        self.resolution = resolution
        self.transforms = transforms
        self.transforms_pose = transforms_pose
        self.vit_resolution = vit_resolution
        self.vit_transforms = vit_transforms
        self.kernel_size = kernel_size # for mask augmentation not used for now
        self.iters = iters # for mask augmentation not used for now
        self.weights = weights # for define the possibility of different masks (random mask or input mask)
        self.turn_background_mask = turn_background_mask

        self.video_list = sorted(glob.glob(data_list + '/*/blender_render/*'))
        self.video_frame_paths = {
            video_path: sorted(glob.glob(os.path.join(video_path, 'stage2', '*')))
            for video_path in self.video_list
        }

        self.sample_fps = sample_fps
        print(self.max_frames, self.sample_fps)
        '''self.seed = 42 
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)'''

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        data_path = self.video_list[index]
        video_key = data_path.split('/')[-3]
        ref_frame, vit_frame, first_frame, video_data, c_video_data, mask, dwpose = self._get_video_data(data_path, video_key)
        return ref_frame, vit_frame, first_frame, video_data, c_video_data, dwpose, mask, video_key
    

    def _process_frames_parallel(self, frame_list, crop_params, angle):
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(lambda frame: self._process_frame(frame, crop_params, angle), frame_list))
        return processed_frames

    def _get_video_data(self, data_path, video_key):
        frame_list = []
        c_frame_list = []
        fg_c_list = []
        dwpose_list = []
        mask_list = []
        no_mask_list = []
        all_paths = self.video_frame_paths[data_path]
        #all_paths = sorted(glob.glob(os.path.join(data_path, 'stage2', '*')))
        if len(all_paths) < self.max_frames * self.sample_fps:
            all_frames = list(range(0, len(all_paths), self.sample_fps))
            frame_indices = all_frames
        else:
            rand_idx = random.randint(0, (len(all_paths) - self.max_frames * self.sample_fps)) 
            all_frames = list(range(rand_idx, len(all_paths), self.sample_fps))
            frame_indices = all_frames[:self.max_frames]
        
        
        # load the ref img
        ref_path = os.path.dirname(os.path.dirname(data_path))
        ref_frame = Image.open(os.path.join(ref_path, 'char/texture.png'))
        ref_dwpose = Image.open(os.path.join(ref_path, 'char/openpose.png'))

        if ref_frame.mode != 'RGB':
            ref_frame = RGBA2RGB(ref_frame)

        if ref_dwpose.mode != 'RGB':
            ref_dwpose = RGBA2RGB(ref_dwpose)

        dwpose_list.append(ref_dwpose)
        input_mask = False
  

        #read the video and dwpose
        for i in frame_indices:
            image = Image.open(all_paths[i])
            coarse_image = Image.open(all_paths[i].replace('stage2', 'color')) #black background
            dwpose = Image.open(all_paths[i].replace('stage2', 'openpose'))
            if os.path.exists(all_paths[i].replace('stage2', 'mask')):
                # 1 for regenerate, 0 for having coarse
                mask = Image.open(all_paths[i].replace('stage2', 'mask'))
                input_mask = True
            else:
                mask = Image.new('RGB', (image.size[0], image.size[1]), 'black')
        
            if image.mode != 'RGB':
                image = RGBA2RGB(image)
            if coarse_image.mode != 'RGB':
                c_alpha = np.array(coarse_image)[:, :, 3]
                fg_coarse = np.where(c_alpha == 255, 1, 0).astype(np.uint8) #check if need>0 or 255
                fg_c_list.append(Image.fromarray(fg_coarse))
                coarse_image = RGBA2RGB(coarse_image)
            if dwpose.mode != 'RGB':
                dwpose = RGBA2RGB(dwpose)
            if mask.mode != 'RGB':
                if self.turn_background_mask:
                    mask = RGBA2RGB(mask, (0,0,0))
                    mask = np.array(mask)
                    mask = np.where(mask == 255, 255, 0).astype(np.uint8) 
                    mask = Image.fromarray(255-mask)
                    mask = mask.convert('L')
                
                else:
                    print('error mask')

            no_mask = Image.new("RGB", (self.resolution[0], self.resolution[1]), "black")
            no_mask = no_mask.convert('L')

            frame_list.append(image) 
            c_frame_list.append(coarse_image)
            dwpose_list.append(dwpose)
            mask_list.append(mask)
            no_mask_list.append(no_mask)

        
        video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        c_video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        dwpose = torch.zeros(self.max_frames+1, 3, self.resolution[1], self.resolution[0])
        mask = torch.zeros(self.max_frames, 1, self.resolution[1], self.resolution[0])

        
        size = [self.resolution[1], self.resolution[0]]
        crop_params = transforms.RandomResizedCrop(size).get_params(ref_frame, scale=(0.7, 1.0), ratio=(0.66, 1.0))
        angle = transforms.RandomRotation(degrees=30).get_params((-30, 30))
      
        
        ref_frame = transforms.functional.resized_crop(ref_frame, *crop_params, size, transforms.InterpolationMode.BILINEAR)
        dwpose_list = [transforms.functional.resized_crop(dwpose_list[i], *crop_params, size, InterpolationMode.BILINEAR) for i in range(len(dwpose_list))]
        fg_c_list = [transforms.functional.resized_crop(fg_c_list[i], *crop_params, size, InterpolationMode.NEAREST) for i in range(len(fg_c_list))]
        frame_list = [transforms.functional.resized_crop(frame_list[i], *crop_params, size, InterpolationMode.BILINEAR) for i in range(len(frame_list))]
        c_frame_list = [transforms.functional.resized_crop(c_frame_list[i], *crop_params, size, InterpolationMode.BILINEAR) for i in range(len(c_frame_list))]

        org_mask_list = None
        if input_mask:
            org_mask_list = [transforms.functional.resized_crop(mask_list[i], *crop_params, size, InterpolationMode.NEAREST) for i in range(len(mask_list))]
        
        mask_list, fill_value = create_random_shape_with_random_motion(len(c_frame_list), imageHeight=self.resolution[1], imageWidth=self.resolution[0], foreground=fg_c_list, org_masks=org_mask_list, turn_background_mask=self.turn_background_mask, weights=self.weights)

        
        filp_p = random.random()
        if filp_p < 0.4:
            dwpose_list = [dwpose_list[0]]+[transforms.functional.vflip(dwpose_list[i]) for i in range(1, len(dwpose_list))]
            dwpose_list = [dwpose_list[0]]+[transforms.functional.rotate(dwpose_list[i], angle, interpolation=InterpolationMode.BICUBIC) for i in range(1,len(dwpose_list))]
            frame_list = [transforms.functional.vflip(frame_list[i]) for i in range(len(frame_list))]
            frame_list = [transforms.functional.rotate(frame_list[i], angle, fill=(255, 255, 255), interpolation=InterpolationMode.BICUBIC) for i in range(len(frame_list))]
            c_frame_list = [transforms.functional.vflip(c_frame_list[i]) for i in range(len(c_frame_list))]
            c_frame_list = [transforms.functional.rotate(c_frame_list[i], angle, fill=(255, 255, 255), interpolation=InterpolationMode.BICUBIC) for i in range(len(c_frame_list))]
            #fg_c_list = [transforms.functional.vflip(fg_c_list[i]) for i in range(len(fg_c_list))]
            #fg_c_list = [transforms.functional.rotate(fg_c_list[i], angle, fill=(0), interpolation=InterpolationMode.NEAREST) for i in range(len(fg_c_list))]
            

        else:
            dwpose_list = [dwpose_list[0]]+[transforms.functional.rotate(dwpose_list[i], angle, interpolation=InterpolationMode.BICUBIC) for i in range(1,len(dwpose_list))]
            frame_list = [transforms.functional.rotate(frame_list[i], angle, fill=(255, 255, 255), interpolation=InterpolationMode.BICUBIC) for i in range(len(frame_list))]
            c_frame_list = [transforms.functional.rotate(c_frame_list[i], angle, fill=(255, 255, 255), interpolation=InterpolationMode.BICUBIC) for i in range(len(c_frame_list))]
            #fg_c_list = [transforms.functional.rotate(fg_c_list[i], angle, fill=(0), interpolation=InterpolationMode.NEAREST) for i in range(len(fg_c_list))]

            
        
        if filp_p < 0.4:
            mask_list = [transforms.functional.vflip(mask_list[i]) for i in range(len(mask_list))]
        if self.turn_background_mask:
            mask_list = [transforms.functional.rotate(mask_list[i], angle, fill=(fill_value), interpolation=InterpolationMode.NEAREST) for i in range(len(mask_list))]
        else:
            mask_list = [transforms.functional.rotate(mask_list[i], angle, fill=(fill_value), interpolation=InterpolationMode.NEAREST) for i in range(len(mask_list))]
        
        mask_p = random.random()
        if mask_p < 0.4:
            num = random.randint(1, len(frame_list))
            mask_list[:num] = [no_mask_list[i] for i in range(num)]

        
        first_frame_tensor = self.transforms_pose(frame_list[0])
        vit_frame = self.vit_transforms(ref_frame)
        ref_frame_tensor = self.transforms(ref_frame)
        frame_tensor = self.transforms(frame_list) #rotate
        dwpose_tensor = self.transforms_pose(dwpose_list) # rotate
        mask_tensor = self.transforms_pose(mask_list) # f, 1, h, w # rotate
        c_frame_tensor = self.transforms_pose(c_frame_list)
        video_data[:len(frame_list), ...] = frame_tensor
        c_video_data[:len(frame_list), ...] = c_frame_tensor
        ref_frame = ref_frame_tensor
        dwpose[:len(frame_list)+1, ...] = dwpose_tensor
        mask[:len(frame_list), ...] = mask_tensor[:,:1,...]
        first_frame = first_frame_tensor

        return ref_frame, vit_frame, first_frame, video_data, c_video_data, mask, dwpose



'''data_list = 'path_to_videos'
dataset = VideoDataset(data_list)
dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

for data in dataloader:
    ref_frame, vit_frame, first_frame, video_data, c_video_data, mask, dwpose = data'''