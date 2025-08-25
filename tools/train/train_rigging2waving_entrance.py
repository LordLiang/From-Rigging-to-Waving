import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import random
import torch
import logging
import datetime
import numpy as np
from PIL import Image
import torch.optim as optim 
from einops import rearrange
import torch.cuda.amp as amp
from importlib import reload
from copy import deepcopy, copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import utils.transforms as data
from utils.util import to_device
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.optim import AnnealingLR
from utils.multi_port import find_free_port
from utils.distributed import generalized_all_gather, all_reduce
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN
from ..modules.unet.util import ResBlock, SpatialTransformer, TemporalTransformer
from tools.modules.autoencoder import get_first_stage_encoding
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@ENGINE.register_function()
def train_rigging2waving_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg


def worker(gpu, cfg):
    '''
    Training worker for each gpu
    '''
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    
    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)
    
    # [Log] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.cfg_file).split('.')[0]
    cfg.log_dir = osp.join(cfg.log_dir, exp_name)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        cfg.log_file = log_file
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
     

    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Dataset] imagedataset and videodataset
    len_frames = len(cfg.frame_lens)
    len_fps = len(cfg.sample_fps)
    cfg.max_frames = cfg.frame_lens[cfg.rank % len_frames]

    
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]
    
    if cfg.rank == 0:
        logging.info(f'Currnt worker with max_frames={cfg.max_frames}, batch_size={cfg.batch_size}, sample_fps={cfg.sample_fps}')
    train_trans = data.Compose([
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)
        ])


    train_trans_pose = data.Compose([
        data.ToTensor(),
        ]
        )
    
    vit_trans = data.Compose([
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)
        ])
    
  
#cfg.num_workers
    dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=vit_trans, 
                             transforms_pose=train_trans_pose, max_frames=cfg.max_frames)
    
    sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True) if (cfg.world_size > 1 and not cfg.debug) else None
    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    rank_iter = iter(dataloader) 
    
    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text="")
    # _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    # zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    # [Model] UNet 
    if "config" in cfg.UNet:
        cfg.UNet["config"] = cfg
    cfg.UNet["zero_y"] = zero_y
    model = MODEL.build(cfg.UNet)
    # load the pretrained checkpoint
    model, resume_step = PRETRAIN.build(cfg.Pretrain, model=model, optimizer=None, scaler=None, resume_from_zero=cfg.resume_from_zero)
    model.requires_grad_(False)
    model = model.to(gpu)
 

    if cfg.train_layers == 'spatial':
        print('train the spatial transformer')
        for name, module in model.named_modules():
            if isinstance(module, SpatialTransformer):
                for params in module.parameters():
                    params.requires_grad = True

    elif cfg.train_layers == 'resblock':
        print('train the resblock')
        for name, module in model.named_modules():
            if isinstance(module, ResBlock):
                for params in module.parameters():
                    params.requires_grad = True

    elif cfg.train_layers == 'temporal':
        print('train the temporal transformer')
        for name, module in model.named_modules():
            if isinstance(module, TemporalTransformer):
                for params in module.parameters():
                    params.requires_grad = True

    else:
        print('no such layers to train')
    
    if cfg.train_pose:
        for param in model.dwpose_embedding.parameters():
            param.requires_grad = True

        for param in model.dwpose_embedding_after.parameters():
            param.requires_grad = True

        for param in model.randomref_pose2_embedding.parameters():
            param.requires_grad = True

        for param in model.randomref_pose2_embedding_after.parameters():
            param.requires_grad = True
        
    for param in model.coarse_image_embedding.parameters():
        param.requires_grad = True
    
    for param in model.coarse_image_embedding_after.parameters():
        param.requires_grad = True


##training parameters include the motion, spatial and coarse embedding

    # optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"Total trainable params {len(trainable_params)}") #1529
    optimizer = optim.AdamW(params=trainable_params,
                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=cfg.use_fp16)

        
    if cfg.use_ema:
        ema = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        ema = type(ema)([(k, ema[k].data.clone()) for k in list(ema.keys())[cfg.rank::cfg.world_size]])
    
    
    if cfg.use_fsdp:
        config = {}
        config['compute_dtype'] = torch.float32
        config['mixed_precision'] = True
        model = FSDP(model, **config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model.to(gpu)

    # scheduler
    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,  # 10
        total_steps=cfg.num_steps,      # 200000
        decay_mode=cfg.decay_mode)      # 'cosine'
    
    # [Visual]
    viz_num = min(cfg.batch_size, 2)
    visual_func = VISUAL.build(
        cfg.visual_train,
        cfg_global=cfg,
        viz_num=viz_num, 
        diffusion=diffusion, 
        guide_scale=cfg.guide_scale, 
        use_offset_noise=None,
        autoencoder=autoencoder)
    
    for step in range(resume_step, cfg.num_steps): 
        model.train()

        try:
            batch = next(rank_iter)
        except StopIteration:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)
        
        batch = to_device(batch, gpu, non_blocking=True)

        ref_frame, vit_frame, _, video_data, c_video_data, dwpose, mask, video_key = batch

        batch_size, frames_num, _, height, width = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        dwpose = rearrange(dwpose, 'b f c h w -> b c f h w')
        mask = rearrange(mask, 'b f c h w -> b c f h w')
        dwpose0 = dwpose.detach().clone()

        # fps_tensor =  torch.tensor([cfg.sample_fps] * batch_size, dtype=torch.long, device=gpu)
        #print(video_data.shape, cfg.chunk_size) #torch.Size([1, 4, 3, 48, 32])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        # vae get the latents
        with torch.no_grad():
            decode_data = []
            for i in range(len(video_data_list)):
                latent_z = autoencoder.encode_firsr_stage(video_data_list[i], cfg.scale_factor).detach()
                decode_data.append(latent_z) # [B, 4, 32, 56]
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size) # [B, 4, 16, 96, 64]

            #ref_frame: b c h w 1 3 768 512
            latent_ref = autoencoder.encode_firsr_stage(ref_frame, cfg.scale_factor).detach()
            latent_ref = latent_ref.unsqueeze(2).repeat(1, 1, frames_num, 1, 1)# b c h w -> b c f h w 1 4 16 96 64
            latent_ref0 = latent_ref.detach().clone()
            

        opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
        t_round = torch.randint(0, opti_timesteps, (batch_size, ), dtype=torch.long, device=gpu) # 8

        # preprocess
        # clip encoder-> embedding image!!! 
        if 'image' in cfg.video_compositions:
            with torch.no_grad():
                vit_frame = vit_frame.squeeze(1) # [1, 3, 224, 224] -> [1, 3, 224, 224]
                y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [1, 1, 1024]
                y_visual0 = y_visual.detach().clone()
        
     
        if 'coarse_image' in cfg.video_compositions:
            c_video_data = rearrange(c_video_data, 'b f c h w -> b c f h w')
            c_video_data_clone = (c_video_data * (1-mask)).detach().clone()
            
            if hasattr(cfg, "latent_coarse_image") and cfg.latent_coarse_image:
                c_video_data = rearrange(c_video_data, 'b c f h w -> (b f) c h w')
                c_data_list = torch.chunk(c_video_data, c_video_data.shape[0]//cfg.chunk_size,dim=0)
                with torch.no_grad():
                    c_data = []
                    for i in range(len(c_data_list)):
                        latent_z = autoencoder.encode_firsr_stage(c_data_list[i].sub(0.5).div_(0.5), cfg.scale_factor).detach()
                        c_data.append(latent_z)
                    c_video_data = torch.cat(c_data,dim=0)
                    c_video_data = rearrange(c_video_data, '(b f) c h w -> b c f h w', b = batch_size)
                    
                    mask = rearrange(mask, 'b c f h w -> (b f) c h w')
                    mask = torch.nn.functional.interpolate(
                            mask, size=(height // 8, width // 8))
                    mask = rearrange(mask, '(b f) c h w -> b c f h w', b = batch_size)

            c_video_data = c_video_data * (1-mask)
            c_video_data = torch.cat([c_video_data, mask], dim=1)
            
        
        # forward
        #'image': clip encoder, 'local image': coarse_video , 'randomref'    'dwpose'                       
        model_kwargs = {'y': None, "local_image": None,  "coarse_image": c_video_data, 'image': y_visual, 
                        'dwpose': dwpose, 'randomref': latent_ref}
        
        
        if cfg.use_fsdp:
            loss = diffusion.loss(x0=video_data, 
                t=t_round, model=model, model_kwargs=model_kwargs,
                use_div_loss=cfg.use_div_loss) 
            loss = loss.mean()
        else:
            with amp.autocast(enabled=cfg.use_fp16):
                loss = diffusion.loss(
                        x0=video_data, 
                        t=t_round, 
                        model=model, 
                        model_kwargs=model_kwargs, 
                        use_div_loss=cfg.use_div_loss,
                        ) # cfg.use_div_loss: False    loss: [80]
                loss = loss.mean()
     
        
        # backward
        if cfg.use_fsdp:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if not cfg.use_fsdp:
            scheduler.step()
        
        # ema update
        if cfg.use_ema:
            temp_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in ema.items():
                v.copy_(temp_state_dict[k].lerp(v, cfg.ema_decay))

        all_reduce(loss)
        loss = loss / cfg.world_size
        
        if cfg.rank == 0 and step % cfg.log_interval == 0: # cfg.log_interval: 100
            logging.info(f'Step: {step}/{cfg.num_steps} Loss: {loss.item():.3f} scale: {scaler.get_scale():.1f} LR: {scheduler.get_lr():.7f}')

        # Visualization
        torch.cuda.empty_cache()
        if step == resume_step or step == cfg.num_steps or step % cfg.viz_interval == 0:
            with torch.no_grad():
                #try:
                visual_kwards = [{
                                    'y': None,
                                    "coarse_image": c_video_data[:viz_num],
                                    "local_image": None,
                                    'image': y_visual0[:viz_num],
                                    'dwpose': dwpose0[:viz_num],
                                    'randomref': latent_ref0[:viz_num],
                                    }, 
                                    {
                                    'y': None,
                                    "coarse_image": None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
                ##ToDo: add the coarse image
                input_kwards = {
                    'model': model, 'video_data': video_data[:viz_num], 'step': step, 'ref_frame': ref_frame[:viz_num], 'dwpose': dwpose0[:viz_num], 'local_image': c_video_data_clone[:viz_num, :3]}
                visual_func.run(visual_kwards=visual_kwards, **input_kwards)
                #except Exception as e:
                #    logging.info(f'Save videos with exception {e}')'''
        
        # Save checkpoint
        if step == cfg.num_steps or step % cfg.save_ckp_interval == 0 or step == resume_step:
            os.makedirs(osp.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
            if cfg.rank == 0:
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                save_dict = {
                    'state_dict': model.module.state_dict() if not cfg.debug else model.state_dict(),
                    'step': step}
                torch.save(save_dict, local_model_path)
        
        # Save ema checkpoint
        if (step % cfg.save_ckp_interval == 0) and step >= cfg.ema_start and cfg.use_ema:
            if cfg.use_ema:
                    local_ema_model_path = osp.join(cfg.log_dir, f'checkpoints/ema_{step:08d}_rank{cfg.rank:04d}.pth')
                    save_dict = {
                        'state_dict': ema.module.state_dict() if hasattr(ema, 'module') else ema,
                        'step': step}
                    torch.save(save_dict, local_ema_model_path)
                    if cfg.rank == 0:
                        logging.info(f'Begin to Save ema model to {local_ema_model_path}')
        
        torch.cuda.empty_cache()
               
       
    
    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
    
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()