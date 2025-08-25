'''
/* 
*Copyright (c) 2021, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

The whole denosing process including three stage:
stage1, our model+no contour ref
stage2, mixing the our and pre trained model + no contour ref -> coarse ref
stage3, redenoising from begining with new ref
'''

import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
import pynvml
import logging
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel
import copy

import utils.transforms as data
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_t2vhigen_video_safe, save_video_multiple_conditions_not_gif_horizontal_3col
from utils.util import SmoothAreaRandomDetection
from tools.modules.autoencoder import get_first_stage_encoding
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION
from copy import copy
import cv2
import torch.nn as nn
from tools.datasets.random_mask import create_random_shape_with_random_motion
from tools.modules.unet.unet_rigging2waving import UNetSD_Rigging2Waving
from scipy.ndimage import binary_dilation, generate_binary_structure
from torch.nn import functional as F

@INFER_ENGINE.register_function()
def inference_rigging2waving_entrance_long(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg, cfg_update)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg

def RGBA2RGB(image, bg_color=(255, 255, 255)):
    background = Image.new("RGB", image.size, bg_color)
    background.paste(image, mask=image.split()[3])
    return background



def load_video_frames(ref_image_path, pose_file_path, train_trans, vit_transforms, train_trans_pose, train_trans_mask, max_frames=32, frame_interval = 1, resolution=[512, 768], vit_resolution=[224, 224], kernel_size=15, iters=10, mask_dilation=False):

    dwpose_all = {}
    frames_all = {}
    coarse_all = {}
    mask_all = {}

    for ii_index in sorted(os.listdir(pose_file_path)):
        if ii_index != "ref_dwpose.png":
            dwpose_all[ii_index] = Image.open(os.path.join(pose_file_path, ii_index))
            frames_all[ii_index] = Image.open(ref_image_path)
            coarse_all[ii_index]  = Image.open(os.path.join(pose_file_path.replace('openpose', 'color'), ii_index))
        
       
            if os.path.exists(pose_file_path.replace('openpose', 'mask')):
                #TODO: get the fg region from coarse input
          
                mask = Image.open(os.path.join(pose_file_path.replace('openpose', 'mask'), ii_index))
                mask = RGBA2RGB(mask, (0,0,0))
                #mask = RGBA2RGB(mask, (255,255,255))
                mask = np.array(mask)
                mask = np.where(mask == 255, 255, 0).astype(np.uint8)  
                if mask_dilation:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask = cv2.erode(mask, kernel, iterations=iters)  
                mask_all[ii_index] = Image.fromarray(255 - mask) 

            else:
                width, height = dwpose_all[ii_index].size
                mask_all[ii_index] = Image.new("RGB", (width, height), "black")
            
    
            
    pose_ref = Image.open(ref_image_path.replace('texture.png', 'openpose.png'))

    # Sample max_frames poses for video generation
    stride = frame_interval
    _total_frame_num = len(frames_all)
    if max_frames == "None":
        max_frames = (_total_frame_num-1)//frame_interval + 1
    cover_frame_num = (stride * (max_frames-1)+1)
    if _total_frame_num < cover_frame_num:
        print('_total_frame_num is smaller than cover_frame_num, the sampled frame interval is changed')
        start_frame = 0   # we set start_frame = 0 because the pose alignment is performed on the first frame
        end_frame = _total_frame_num
        stride = max((_total_frame_num-1//(max_frames-1)),1)
        end_frame = stride*max_frames
    else:
        start_frame = 0  # we set start_frame = 0 because the pose alignment is performed on the first frame
        end_frame = start_frame + cover_frame_num


    frame_list = []
    dwpose_list = []
    coarse_list = []
    mask_list = []
    random_ref_frame = frames_all[list(frames_all.keys())[0]]
    if random_ref_frame.mode != 'RGB':
        random_ref_frame = RGBA2RGB(random_ref_frame)
    random_ref_dwpose = pose_ref
    if random_ref_dwpose.mode != 'RGB':
        random_ref_dwpose = RGBA2RGB(random_ref_dwpose)


    for i_index in range(start_frame, end_frame, stride):
        if i_index < len(frames_all):  # Check index within bounds
            i_key = list(frames_all.keys())[i_index]
            i_frame = frames_all[i_key]
            if i_frame.mode != 'RGB':
                i_frame = RGBA2RGB(i_frame)
    
            
            i_dwpose = dwpose_all[i_key]
            if i_dwpose.mode != 'RGB':
                i_dwpose = RGBA2RGB(i_dwpose)
            
            i_coarse = coarse_all[i_key]
            if i_coarse.mode != 'RGB':
                i_coarse = RGBA2RGB(i_coarse)

            
            
            i_mask = mask_all[i_key]
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)
            coarse_list.append(i_coarse)
            mask_list.append(i_mask)
    
    

    if frame_list:

        middle_indix = 0
        ref_frame = frame_list[middle_indix]
        vit_frame = vit_transforms(ref_frame)
        random_ref_frame_tmp = train_trans_pose(random_ref_frame)
        random_ref_dwpose_tmp = train_trans_mask(random_ref_dwpose)
        
        misc_data_tmp = torch.stack([train_trans_pose(ss) for ss in frame_list], dim=0)
        video_data_tmp = torch.stack([train_trans(ss) for ss in frame_list], dim=0)
        dwpose_data_tmp = torch.stack([train_trans_mask(ss) for ss in dwpose_list], dim=0)
        coarse_data_tmp = torch.stack([train_trans_pose(ss) for ss in coarse_list], dim=0)
        mask_data_tmp = torch.stack([train_trans_mask(ss) for ss in mask_list], dim=0)

        video_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
        dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
        misc_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
        coarse_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
        mask_data = torch.zeros(max_frames, 1, resolution[1], resolution[0])
        random_ref_frame_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
        random_ref_dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])

        video_data[:len(frame_list), ...] = video_data_tmp
        misc_data[:len(frame_list), ...] = misc_data_tmp
        dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
       # coarse_data[:len(frame_list), ...] = coarse_data_tmp * (1-mask_data_tmp)
        coarse_data[:len(frame_list), ...] = coarse_data_tmp
        mask_data[:len(frame_list), ...] = mask_data_tmp[:,:1]
        random_ref_frame_data[:, ...] = random_ref_frame_tmp
        random_ref_dwpose_data[:, ...] = random_ref_dwpose_tmp

        return vit_frame, video_data, misc_data, dwpose_data, coarse_data, random_ref_frame_data, random_ref_dwpose_data, mask_data, max_frames


def worker(gpu, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
            torch.backends.cudnn.benchmark = False
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    tuned_model = osp.basename(cfg.tuned_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (inf_name))
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    cfg.log_file = log_file
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    logging.info(f"Running UniAnimate inference on gpu {gpu}")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    train_trans_pose = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        ]
        )
    
    train_trans_mask = data.Compose([
        data.Resize(cfg.resolution, interpolation=Image.NEAREST),
        data.ToTensor(),
        ]
        )

    vit_transforms = T.Compose([
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    with torch.no_grad():
        _, _, zero_y = clip_encoder(text="")
    

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    if "config" in cfg.UNet_pre:
        cfg.UNet_pre["config"] = cfg
        cfg.video_compositions.append('coarse_image')
        cfg.UNet_tune["config"] = cfg
    cfg.UNet_pre["zero_y"] = zero_y
    cfg.UNet_tune["zero_y"] = zero_y

    model_pre = MODEL.build(cfg.UNet_pre)
    model_tune = MODEL.build(cfg.UNet_tune)

    state_dict_pre = torch.load(cfg.pretrained_model, map_location='cpu')
    if 'state_dict' in state_dict_pre:
        state_dict_pre = state_dict_pre['state_dict']
    status = model_pre.load_state_dict(state_dict_pre, strict=False)
    logging.info('Load pretrained model from {} with status {}'.format(cfg.pretrained_model, status))
    del state_dict_pre

    state_dict_tune = torch.load(cfg.tuned_model, map_location='cpu')
    if 'state_dict' in state_dict_tune:
        state_dict_tune = state_dict_tune['state_dict']
    status = model_tune.load_state_dict(state_dict_tune, strict=False)
    logging.info('Load tuned model from {} with status {}'.format(cfg.tuned_model, status))
    del state_dict_tune

    model_pre = model_pre.to(gpu)
    model_tune = model_tune.to(gpu)

    model_pre.eval()
    model_tune.eval()
    if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
        model_pre.to(torch.float16) 
        model_tune.to(torch.float16) 
    #else:
    #    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()


    test_list = cfg.test_list_path
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for _ in range(cfg.round) for item in test_list]

   
    for idx, file_path in enumerate(test_list):
        cfg.frame_interval, ref_image_key, pose_seq_key = file_path[0], file_path[1], file_path[2]
        manual_seed = int(cfg.seed + cfg.rank + idx//num_videos)
        setup_seed(manual_seed)
        logging.info(f"[{idx}]/[{len(test_list)}] Begin to sample {ref_image_key}, pose sequence from {pose_seq_key} init seed {manual_seed} ...")
        
        vit_frame, _, misc_data, dwpose_data, coarse_data, random_ref_frame_data, random_ref_dwpose_data, mask_data, max_frames = load_video_frames(ref_image_key, 
                                                                                                                                                                   pose_seq_key, train_trans, vit_transforms, train_trans_pose, train_trans_mask, max_frames=cfg.max_frames, frame_interval =cfg.frame_interval, resolution=cfg.resolution, mask_dilation=cfg.mask_dilation,  kernel_size=cfg.kernel_size)
        vit_frame_nc, _, misc_data_nc, _, _, random_ref_frame_data_nc, _, _, _ = load_video_frames(ref_image_key.replace('texture', 'ffc_resnet_inpainted'), 
                                                                                                                                                                   pose_seq_key, train_trans, vit_transforms, train_trans_pose, train_trans_mask, max_frames=cfg.max_frames, frame_interval =cfg.frame_interval, resolution=cfg.resolution, mask_dilation=cfg.mask_dilation,  kernel_size=cfg.kernel_size)
        cfg.max_frames_new = max_frames

        misc_data = misc_data.unsqueeze(0).to(gpu)
        vit_frame = vit_frame.unsqueeze(0).to(gpu)
        dwpose_data = dwpose_data.unsqueeze(0).to(gpu)
        coarse_data = coarse_data.unsqueeze(0).to(gpu)
        mask_data = mask_data.unsqueeze(0).to(gpu)
        random_ref_frame_data = random_ref_frame_data.unsqueeze(0).to(gpu) # b f c h w
        random_ref_dwpose_data = random_ref_dwpose_data.unsqueeze(0).to(gpu)

        #referece image without contour
        vit_frame_nc = vit_frame_nc.unsqueeze(0).to(gpu)
        misc_data_nc = misc_data_nc.unsqueeze(0).to(gpu)
        random_ref_frame_data_nc = random_ref_frame_data_nc.unsqueeze(0).to(gpu) # b f c h w


        ### save for visualization
        misc_backups = copy(misc_data)
        misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')

        ### coarse image 
        image_coarse = []
        if 'coarse_image' in cfg.video_compositions:
            coarse_data = rearrange(coarse_data, 'b f c h w -> b c f h w')
            mask_data = rearrange(mask_data, 'b f c h w -> b c f h w')
            image_coarse_clone = (coarse_data).detach().clone()
            masked_image_coarse_clone = (coarse_data * (1 - mask_data)).detach().clone()

            batch_size, _, frames_num, height, width = coarse_data.shape
            latent_coarse_data = rearrange(coarse_data, 'b c f h w -> (b f) c h w')
            c_data_list = torch.chunk(latent_coarse_data, latent_coarse_data.shape[0]//cfg.chunk_size, dim=0)
            with torch.no_grad():
                c_data = []
                for i in range(len(c_data_list)):
                    latent_z = autoencoder.encode_firsr_stage(c_data_list[i].sub(0.5).div_(0.5), cfg.scale_factor).detach()
                    c_data.append(latent_z)
                latent_coarse_data = torch.cat(c_data,dim=0)
                latent_coarse_data = rearrange(latent_coarse_data, '(b f) c h w -> b c f h w', b = batch_size, f=frames_num)
            
            latent_mask_data = rearrange(mask_data, 'b c f h w -> (b f) c h w')
            latent_mask_data = torch.nn.functional.interpolate(
                    latent_mask_data, size=(height // 8, width // 8), mode='nearest')
            latent_mask_data = rearrange(latent_mask_data, '(b f) c h w -> b c f h w', b = batch_size, f=frames_num)
            

            if hasattr(cfg, "latent_coarse_image") and cfg.latent_coarse_image:
                print('using features')
                coarse_data = latent_coarse_data
                mask_data = latent_mask_data

            coarse_data = coarse_data * (1 - mask_data)
            image_coarse = torch.cat([coarse_data, mask_data], dim=1)
        

        with torch.no_grad():

            random_ref_frame = []
            if 'randomref' in cfg.video_compositions:
                random_ref_frame_clone = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')
                if hasattr(cfg, "latent_random_ref") and cfg.latent_random_ref:
                    temporal_length = random_ref_frame_data.shape[1]
                    encoder_posterior = autoencoder.encode(random_ref_frame_data[:,0].sub(0.5).div_(0.5))
                    random_ref_frame_data = get_first_stage_encoding(encoder_posterior).detach()
                    random_ref_frame_data = random_ref_frame_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

                    encoder_posterior = autoencoder.encode(random_ref_frame_data_nc[:,0].sub(0.5).div_(0.5))
                    random_ref_frame_data_nc = get_first_stage_encoding(encoder_posterior).detach()
                    random_ref_frame_data_nc = random_ref_frame_data_nc.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

                random_ref_frame = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')
                random_ref_frame_nc = rearrange(random_ref_frame_data_nc, 'b f c h w -> b c f h w')


            if 'dwpose' in cfg.video_compositions:
                bs_vd_local = dwpose_data.shape[0]
                dwpose_data_clone = rearrange(dwpose_data.clone(), 'b f c h w -> b c f h w', b = bs_vd_local)
                if 'randomref_pose' in cfg.video_compositions:
                    dwpose_data = torch.cat([random_ref_dwpose_data[:,:1], dwpose_data], dim=1)
                dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)

            
            y_visual = []
            if 'image' in cfg.video_compositions:
                with torch.no_grad():
                    vit_frame = vit_frame.squeeze(1)
                    y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                    y_visual0 = y_visual.clone()

                    vit_frame_nc = vit_frame_nc.squeeze(1)
                    y_visual_nc = clip_encoder.encode_image(vit_frame_nc).unsqueeze(1) # [60, 1024]
                    y_visual0_nc = y_visual_nc.clone()


        with amp.autocast(enabled=True):
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            cur_seed = torch.initial_seed()
            logging.info(f"Current seed {cur_seed} ...")

            noise = torch.randn([1, 4, cfg.max_frames_new, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
            noise = noise.to(gpu)
            
            if hasattr(cfg.Diffusion, "noise_strength"):
                b, c, f, _, _= noise.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
                noise = noise + cfg.Diffusion.noise_strength * offset_noise
            
            # add a noise prior
            if cfg.add_noise_prior:
                #noise = diffusion.q_sample(random_ref_frame.clone(), getattr(cfg, "noise_prior_value", 949), noise=noise)
                noise = diffusion.q_sample(latent_coarse_data.clone(), getattr(cfg, "noise_prior_value", 949), noise=noise)

            
            # construct model inputs (CFG)
            # for tuned model with ref_no_contour
            full_model_kwargs_tune_nc=[{
                                    'y': None,
                                    'local_image': None,
                                    "coarse_image": None if len(image_coarse) == 0 else image_coarse[:],
                                    'image': None if len(y_visual_nc) == 0 else y_visual0_nc[:],
                                    'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                                    'randomref': None if len(random_ref_frame_nc) == 0 else random_ref_frame_nc[:],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'coarse_image': None,
                                    'image': None,
                                    'randomref': None,
                                    'dwpose':  None,
                                    }]
            
            
            # for pretrained model with ref_no_contour
            full_model_kwargs_pre_nc=[{
                                    'y': None,
                                    'local_image': None,
                                    "coarse_image": None,
                                    'image': None if len(y_visual_nc) == 0 else y_visual0_nc[:],
                                    'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                                    'randomref': None if len(random_ref_frame_nc) == 0 else random_ref_frame_nc[:],
                                    }, 
                                    {
                                    'y': None,
                                    'coarse_image': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose':  None,
                                    }]
            
            # for our model with ref_contour
            full_model_kwargs_tune_c=[{
                                    'y': None,
                                    'local_image': None,
                                    "coarse_image": None if len(image_coarse) == 0 else image_coarse[:],
                                    'image': None if len(y_visual) == 0 else y_visual0[:],
                                    'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                                    'randomref': None if len(random_ref_frame) == 0 else random_ref_frame[:],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'coarse_image': None,
                                    'image': None,
                                    'randomref': None,
                                    'dwpose':  None,
                                    }]

            # for visualization
            full_model_kwargs_vis =[{
                                    'y': None,
                                    'local_image': None,
                                    "coarse_image": None if len(image_coarse) == 0 else masked_image_coarse_clone[:, :3],
                                    'image': None,
                                    'dwpose': None if len(dwpose_data_clone) == 0 else dwpose_data_clone[:],
                                    'randomref': None if len(random_ref_frame) == 0 else random_ref_frame_clone[:, :3],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    "coarse_image": None,
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]

            '''partial_keys = [
                    ['image', 'randomref', 'local_image', 'coarse_image', 'dwpose'],
                ]'''
            partial_keys = [
                    ['image', 'randomref', 'coarse_image', 'dwpose'],
                ]

            
            if hasattr(cfg, "partial_keys") and cfg.partial_keys:
                partial_keys = cfg.partial_keys
                
            for partial_keys_one in partial_keys:
                model_kwargs_one_tune_nc = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs_tune_nc,
                                    use_fps_condition = cfg.use_fps_condition)
                
                model_kwargs_one_pre_nc = prepare_model_kwargs(partial_keys = partial_keys_one,
                                        full_model_kwargs = full_model_kwargs_pre_nc,
                                        use_fps_condition = cfg.use_fps_condition)
              
                model_kwargs_one_tune_c = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs_tune_c,
                                    use_fps_condition = cfg.use_fps_condition)
                

                model_kwargs_one_vis = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs_vis,
                                    use_fps_condition = cfg.use_fps_condition)
                
                

                noise_one = noise

                if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                    clip_encoder.cpu() 
                    autoencoder.cpu() 
                    torch.cuda.empty_cache() 
                
                model_kwargs_one_pre_nc = [{k: v for k, v in model_kwargs_one_pre_nc[i].items() if k != 'coarse_image'} for i in range(2)]


                video_data = diffusion.ddim_sample_blend_loop_long(
                    noise=noise_one,
                    context_size=cfg.context_size, 
                    context_overlap=cfg.context_overlap,
                    models=[model_pre, model_tune], 
                    model_kwargs=[model_kwargs_one_tune_nc, model_kwargs_one_pre_nc, model_kwargs_one_tune_c],
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0,
                    context_batch_size=getattr(cfg, "context_batch_size", 1),
                    mask=latent_mask_data,
                    vae=autoencoder,
                    tau_alpha=cfg.tau_alpha,
                    tau_beta=cfg.tau_beta,
                    coarse_nomask = image_coarse_clone,
                    )
            
                #video_data = noise
                if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                    # if run forward of  autoencoder or clip_encoder second times, load them again
                    clip_encoder.cuda()
                    autoencoder.cuda()
                video_data = 1. / cfg.scale_factor * video_data 
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                chunk_size = min(cfg.decoder_bs, video_data.shape[0])
                video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
                decode_data = []
                for vd_data in video_data_list:
                    gen_frames = autoencoder.decode(vd_data)
                    decode_data.append(gen_frames)
                video_data = torch.cat(decode_data, dim=0)

                if cfg.resize:
                    video_data = F.interpolate(video_data, size=(512, 512), mode='bilinear', align_corners=False)

                
                video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size).float()

                
                text_size = cfg.resolution[-1]
                cap_name = re.sub(r'[^\w\s]', '', ref_image_key.split("/")[-3]) # .replace(' ', '_')
                name = f'seed_{cur_seed}'
                for ii in partial_keys_one:
                    name = name + "_" + ii
                file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:02d}_{name}_{cap_name}_{cfg.resolution[1]}x{cfg.resolution[0]}.mp4'
                local_path = os.path.join(cfg.log_dir, f'{file_name}')
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
          
                del model_kwargs_one_vis[0][list(model_kwargs_one_vis[0].keys())[0]]
                del model_kwargs_one_vis[1][list(model_kwargs_one_vis[1].keys())[0]]

                save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_data.cpu(), model_kwargs_one_vis, misc_backups, 
                                                cfg.mean, cfg.std, nrow=1, save_fps=cfg.save_fps)
                torch.cuda.empty_cache()
                
                # try:
                #     save_t2vhigen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
                #     logging.info('Save video to dir %s:' % (local_path))
                # except Exception as e:
                #     logging.info(f'Step: save text or video error with {e}')
    
    logging.info('Congratulations! The inference is completed!')
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs