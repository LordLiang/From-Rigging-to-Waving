import glob

import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
import yaml

import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np
import xformers
import open_clip
import xformers.ops

import logging
logger = logging.getLogger(__name__)

# Modified from tokenflow_utils.py
def register_time(model, t):
    #conv_module = model.unet.output_blocks[1].resnets[1]
    #setattr(conv_module, "t", t)
    #up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [1, 2]}

    res_dict = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        module = model.output_blocks[res][1].transformer_blocks[0].attn1
        setattr(module, "t", t)
        module = model.output_blocks[res][1].transformer_blocks[0].attn2
        setattr(module, "t", t)
        #module = model.output_blocks[res][2].transformer_blocks[0].attn1
        #setattr(module, "t", t)
        #module = model.output_blocks[res][2].transformer_blocks[0].attn2
        #setattr(module, "t", t)
    res_dict = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    for res in res_dict:
        module = model.output_blocks[res][2].transformer_blocks[0].attn1
        setattr(module, "t", t)
        #module = model.output_blocks[res][2].transformer_blocks[0].attn2
        #setattr(module, "t", t)



# Modified from models/attention.py
from typing import Optional
from .util import MemoryEfficientCrossAttention, default, exists

def register_self_attention_pnp(model, injection_schedule):
    #class ModifiedSelfAttn(MemoryEfficientCrossAttention):
    def modifiedselfattn(self, x, context=None, mask=None, q_s=None, k_s=None, mask_s=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        # Modified here
        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            logger.debug(f"PnP Injecting Self-Attn at t={self.t}")
            # inject source into unconditional/conditional
            q = torch.cat([q_s, q], dim=0)
            k = torch.cat([k_s, k], dim=0)
            v = torch.cat([v, v], dim=0)

    
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if q.shape[0] > self.max_bs:
            q_list = torch.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = torch.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = torch.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                out = xformers.ops.memory_efficient_attention(
                    q_1, k_1, v_1, attn_bias=None, op=self.attention_op)
                out_list.append(out)
            out = torch.cat(out_list, dim=0)
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            out_s, out_f = torch.chunk(out, b//2, dim=0)
            out = out_s * mask_s + out_f * (1 - mask_s)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

    def create_forward_with_extra_inputs(module, q_s, k_s, mask_s):
        def forward_with_extra_inputs(x, context=None, mask=None):
            return modifiedselfattn(module, x, context=context, q_s=q_s, k_s=k_s, mask_s=mask_s)
        return forward_with_extra_inputs
    
    res_dict = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        module = model.output_blocks[res][1].transformer_blocks[0].attn1
        #module.forward = modifiedselfattn.__get__(module) #input the query, and key from the reference
        q_s, k_s, mask_s = None, None, None
        module.forward = create_forward_with_extra_inputs(module, q_s, k_s, mask_s)
        setattr(module, "injection_schedule", injection_schedule)
        #modified_processor = ModifiedSelfAttn()
        
        #module.processor = modified_processor



def register_cross_attention_pnp(model, injection_schedule):
  
    def modifiedcrossattn(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        # Modified here
        chunk_size = b // 2  # batch_size is 2*chunk_size because concat[source, uncond] or [source, cond]
        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            logger.debug(f"PnP Injecting Self-Attn at t={self.t}")
            # inject source into unconditional/conditional
            q[chunk_size : 2 * chunk_size] = q[:chunk_size]
            k[chunk_size : 2 * chunk_size] = k[:chunk_size]
    

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if q.shape[0] > self.max_bs:
            q_list = torch.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = torch.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = torch.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                out = xformers.ops.memory_efficient_attention(
                    q_1, k_1, v_1, attn_bias=None, op=self.attention_op)
                out_list.append(out)
            out = torch.cat(out_list, dim=0)
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)
    
    res_dict = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        module = model.output_blocks[res][1].transformer_blocks[0].attn2
        module.forward = modifiedcrossattn.__get__(module)
        setattr(module, "injection_schedule", injection_schedule)

def register_temp_attention_pnp(model, injection_schedule):
    def modifiedtempattn(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        # Modified here
        chunk_size = b // 2  # batch_size is 2*chunk_size because concat[source, uncond] or [source, cond]
        if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            logger.debug(f"PnP Injecting Self-Attn at t={self.t}")
            # inject source into unconditional/conditional
            q[chunk_size : 2 * chunk_size] = q[:chunk_size]
            k[chunk_size : 2 * chunk_size] = k[:chunk_size]
    

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if q.shape[0] > self.max_bs:
            q_list = torch.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = torch.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = torch.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                out = xformers.ops.memory_efficient_attention(
                    q_1, k_1, v_1, attn_bias=None, op=self.attention_op)
                out_list.append(out)
            out = torch.cat(out_list, dim=0)
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)
    
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         module.attn1.processor.__call__ = sa_processor__call__(module.attn1.processor)
    #         setattr(module.attn1.processor, "injection_schedule", [])  # Disable PNP

    res_dict = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    #res_dict = [7, 8, 9, 10, 11]
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        module = model.output_blocks[res][2].transformer_blocks[0].attn1
        module.forward = modifiedtempattn.__get__(module)
        setattr(module, "injection_schedule", injection_schedule)
       # module = model.output_blocks[res][2].transformer_blocks[0].attn2
       # module.forward = modifiedtempattn.__get__(module)
       # setattr(module, "injection_schedule", injection_schedule)