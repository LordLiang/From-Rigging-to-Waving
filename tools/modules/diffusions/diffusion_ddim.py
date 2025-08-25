import torch
import math
from torch.optim.adam import Adam
import torch.nn.functional as nnf
import imageio
import os
import cv2
from utils.registry_class import DIFFUSION
from .schedules import beta_schedule, sigma_schedule
from .losses import kl_divergence, discretized_gaussian_log_likelihood, poisson_blend
# from .dpm_solver import NoiseScheduleVP, model_wrapper_guided_diffusion, model_wrapper, DPM_Solver
from typing import Callable, List, Optional
import numpy as np
from ..unet.pnp_utils import register_time
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from ..unet.util import ResBlock, SpatialTransformer, TemporalTransformer, Downsample, Upsample
import torchvision.transforms as transforms

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    self.scheduler.alphas_cumprod[timestep]
    """
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)

@DIFFUSION.register_class()
class DiffusionDDIMSR(object):
    def __init__(self, reverse_diffusion, forward_diffusion, **kwargs):
        from .diffusion_gauss import GaussianDiffusion
        self.reverse_diffusion = GaussianDiffusion(sigmas=sigma_schedule(reverse_diffusion.schedule, **reverse_diffusion.schedule_param), 
                                                   prediction_type=reverse_diffusion.mean_type)
        self.forward_diffusion = GaussianDiffusion(sigmas=sigma_schedule(forward_diffusion.schedule, **forward_diffusion.schedule_param), 
                                                   prediction_type=forward_diffusion.mean_type)


@DIFFUSION.register_class()
class DiffusionDPM(object):
    def __init__(self, forward_diffusion, **kwargs):
        from .diffusion_gauss import GaussianDiffusion
        self.forward_diffusion = GaussianDiffusion(sigmas=sigma_schedule(forward_diffusion.schedule, **forward_diffusion.schedule_param),
            prediction_type=forward_diffusion.mean_type) 


@DIFFUSION.register_class()
class DiffusionDDIM(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 **kwargs):
        
        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # eps
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength # 0.0

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise


    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) # guide_scale=9.0
        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)
        
        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'v':
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0
    
    @torch.no_grad()
    def ddim_inversion(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input, guidance_scale=None
        b = x0.size(0)
        xt = x0
        all_latent = [xt]

        # reconstruction steps -> ddim_timesteps+1
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        print(steps)
        for i, step in enumerate(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, xt, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
            all_latent.append(xt)
        return all_latent

   
    @torch.no_grad()
    def ddim_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, vae=None):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)
            
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0


    @torch.no_grad()
    def visualize_x(self, vae, x, t, name):
        vae.cuda()
        coarse_I = 1. / 0.18215 * x
        b, c, f, h, w = coarse_I.shape
        coarse_I = rearrange(coarse_I, 'b c f h w -> (b f) c h w') 
        coarse_I_list = torch.chunk(coarse_I, 8, dim=0)
        decode_data = []
        for vd_data in coarse_I_list:
            gen_frames = vae.decode(vd_data.float())
            decode_data.append(gen_frames)
        coarse_I = torch.cat(decode_data, dim=0)
        coarse_I = rearrange(coarse_I, '(b f) c h w -> b c f h w', b=b, f=f) 
        coarse_I = (coarse_I.clamp(-1,1) + 1) / 2
        frames = []
        for i in range(coarse_I.shape[2]):
            frame = coarse_I[0][:3,i].permute(1, 2, 0).cpu().numpy()  
            frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        os.makedirs('visualization', exist_ok=True)
        imageio.mimsave(os.path.join('visualization', str(t) + '_' + name + '.gif'), frames, duration=0.1)
    

    @torch.no_grad()
    def ddim_sample_condition(self, xt, t, models, model_kwargs, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, mask=None, vae=None, tau_start=0, tau_stop=0, coarse_nomask=None):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps

        model_pre, model_tune = models
        # predict distribution of p(x_{t-1} | x_t)
        coarse_I = None
        if t >= tau_start:
            _, _, _, x0_our = self.p_mean_variance(xt, t, model_tune, model_kwargs[0], clamp, percentile, guide_scale)
            x0 = x0_our
            #self.visualize_x(vae, x0, t, 'now')

        else:
            _, _, _, x0_our = self.p_mean_variance(xt, t, model_tune, model_kwargs[0], clamp, percentile, guide_scale)
            #self.visualize_x(vae, x0_our, t, 'our')
            _, _, _, x0_pre = self.p_mean_variance(xt, t, model_pre, model_kwargs[1], clamp, percentile, guide_scale)
            #self.visualize_x(vae, x0_pre, t, 'pre')
            x0 = (1 - mask) * x0_our +  mask * x0_pre
            #self.visualize_x(vae, x0, t, 'mix')
    
            if t == tau_stop:
                print('update the coarse_condition:', t)
                #get the orginal coarse and mask inputs
                coarse_org, mask_org = model_kwargs[0][0]['coarse_image'][:,:3], model_kwargs[0][0]['coarse_image'][:,3:]
                # update the mask conditions
                mask_new = torch.zeros_like(mask_org)
                #print(torch.unique(mask_org))

                # decode the x0
                vae.cuda()
                coarse_I = 1. / 0.18215 * x0
                b, _, f, _, _ = coarse_I.shape
                coarse_I = rearrange(coarse_I, 'b c f h w -> (b f) c h w') 

                coarse_I_list = torch.chunk(coarse_I, 8, dim=0)
                decode_data = []
                for vd_data in coarse_I_list:
                    gen_frames = vae.decode(vd_data.float())
                    decode_data.append(gen_frames)
                coarse_I = torch.cat(decode_data, dim = 0)
                coarse_I = rearrange(coarse_I, '(b f) c h w -> b c f h w', b = b, f = f)
                coarse_I = ((coarse_I + 1) / 2).clamp(0,1) 

                
                # blend the coarse_I and coarse_org with Poisson Blending
                #coarse_I = mask_org * coarse_I + (1 - mask_org) * coarse_org 
                '''frames = []
                for i in range(coarse_I.shape[2]):
                    frame = coarse_I_concate[0][:3,i].permute(1, 2, 0).cpu().numpy()  
                    frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)
                imageio.mimsave(str(t)+'_coarse_concate.gif', frames, duration=0.1)'''

                coarse_I = rearrange(coarse_I, 'b c f h w -> (b f) h w c') 
                mask_org = rearrange(mask_org, 'b c f h w -> (b f) h w c') 
                coarse_org = rearrange(coarse_org, 'b c f h w -> (b f) h w c') 
                if coarse_nomask is not None:
                    coarse_nomask = rearrange(coarse_nomask, 'b c f h w -> (b f) h w c') 
                else:
                    print('input masked coarse')
                    coarse_nomask = coarse_org
                mask = 1 - mask_org

                blended_frames = []
                for i in range(coarse_I.shape[0]): 
                    blended_frame = poisson_blend(coarse_nomask[i], coarse_I[i], mask[i])
                    blended_frames.append(blended_frame.unsqueeze(0))
                blended_frames = torch.concat(blended_frames, dim=0)
                coarse_I = rearrange(blended_frames, '(b f) h w c -> b c f h w', b = b, f = f) #(-1,1)'''

                # update the coarse conditions
                coarse_I = coarse_I.to(mask_new.device).to(mask_new.dtype)
                coarse_I = torch.cat([coarse_I, mask_new], dim=1)
                model_kwargs[0][0]['coarse_image'] = coarse_I
                model_kwargs[2][0]['coarse_image'] = coarse_I

                # for visualization coarse result
                coarse_I = model_kwargs[2][0]['coarse_image'] # b c f h w
                frames = []
                for i in range(coarse_I.shape[2]):
                    frame = coarse_I[0][:3,i].permute(1, 2, 0).cpu().numpy()  
                    frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)
                imageio.mimsave(str(t)+'_coarse_possion.gif', frames, duration=0.1)
                

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask_noise = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_sdi = torch.sqrt(alphas_prev) * x0 + direction + mask_noise * sigmas * noise

        return xt_sdi, x0


    @torch.no_grad()
    def ddim_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, vae=None, return_coarse=False):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        #steps = (torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        #print(steps)
        from tqdm import tqdm
        for step in tqdm(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, x0 = self.ddim_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta, vae)
            #self.visualize_x(vae, x0, step, 'x0')
            # from ipdb import set_trace; set_trace()
        if return_coarse:
            return xt, x0
        return xt
    


    @torch.no_grad()
    def ddim_sample_blend_loop(self, noise, models, model_kwargs, clamp=None, percentile=None, condition_fn=None, 
                               guide_scale=None, ddim_timesteps=20, eta=0.0, mask=None, vae=None, tau_alpha=0.9, tau_beta=0.8, return_coarse=False, coarse_nomask=None):
        # prepare input
        b = noise.size(0)
        xt_sdi = noise
        #self.visualize_x(vae, xt1, 949, 'x0')
        xt = noise.clone()

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)

        # get the start and stop step for the secondary motion injection
        if tau_alpha > 0 and tau_alpha > tau_beta:
            tau_start, tau_stop = steps[-(int(ddim_timesteps * tau_alpha) + 1)], steps[-(int(ddim_timesteps * tau_beta) + 1)]
            print('tau_start:', tau_start, 'tau_stop:', tau_stop)
        else:
            tau_start, tau_stop = 0, 0 # without the sencodary motion injection

        from tqdm import tqdm
        if tau_start > 0:
            print('with the sencodary motion injection')
            for i, step in enumerate(tqdm(steps)):
                t = torch.full((b, ), step, dtype=torch.long, device=xt_sdi.device)
                xt_sdi, x0_coarse = self.ddim_sample_condition(xt_sdi, t, models, [model_kwargs[0], model_kwargs[1], model_kwargs[2]], clamp, 
                                                percentile, condition_fn, 1.5, ddim_timesteps, eta, mask, vae, tau_start, tau_stop, coarse_nomask)
                #self.visualize_x(vae, xt1, t, 'x0')
                if step == tau_stop:
                    break
        
            for i, step in enumerate(tqdm(steps)):
                t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
                xt, x0 = self.ddim_sample(xt, t, models[1], model_kwargs[2], clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
        else:
            print('without the sencodary motion injection')
            for i, step in enumerate(tqdm(steps)):
                t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
                xt, x0 = self.ddim_sample(xt, t, models[1], model_kwargs[2], clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
            x0_coarse = x0

        if return_coarse:
            return xt, x0_coarse, x0
        else:
            return xt
        

    @torch.no_grad()
    def transfer2image(self, latent, vae):
        vae.cuda()
        latent = 1. / 0.18215   * latent
        b, c, f, h, w= latent.shape
        latent = rearrange(latent, 'b c f h w -> (b f) c h w') #16,c,h,w

        latent_list = torch.chunk(latent, 8, dim=0)
        decode_data = []
        for vd_data in latent_list:
            gen_frames = vae.decode(vd_data.float())
            decode_data.append(gen_frames)
        image = torch.cat(decode_data, dim=0)
        image = rearrange(image, '(b f) c h w -> b c f h w', b=b, f=f) #(-1,1)
        image = (image.clamp(-1,1) + 1)/2 #(0,1)
        return image
    
    @torch.no_grad()
    def ddim_sample_blend_loop_long(self, noise, context_size, context_overlap, models, model_kwargs, clamp=None, percentile=None, 
                                condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_batch_size=1, mask=None, vae=None, tau_alpha=0.1, tau_beta=0.2, coarse_nomask=None):
  

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        #steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        from tqdm import tqdm
        # divde clip
        bs_context = 1
        
        total_frames = noise.shape[2]
        video_data = []
        #keep_first = False
        current_start = 0
        coarse_seg = None

        counter = torch.zeros(
            (1, 1, total_frames, 1, 1),
            device=noise.device,
            dtype=noise.dtype,
        )

        video_data_all = torch.zeros_like(noise)

        while current_start < total_frames:
            # 计算当前clip的结束帧
            current_end = current_start + context_size
            if current_end >  total_frames:
                if current_start == 0:
                    current_end = total_frames
                else:
                    remaining_frames =  total_frames - current_start
                    current_start = current_start - (context_size - remaining_frames)
                    context_overlap = context_overlap + (context_size - remaining_frames)
                    current_end =  total_frames
            
            
            context = list(range(current_start, current_end))
   
            ind = torch.tensor(context)
            pose_ind = torch.tensor([0] + [i+1 for i in context])
            noise_one = noise[:, :, ind, :, :]
            mask_seg = mask[:, :, ind, :, :]
            coarse_nomask_seg = coarse_nomask[:, :, ind, :, :]

            if current_start > 0:
                # Overwrite the first frame of the new segment with the last frame of the previous video
                # transfer coarse_seg_noc and coarse_seg -> image
                coarse_seg_noc = model_kwargs_tune_c[0]['coarse_image'].clone()
            


            model_kwargs_tune_nc = [{
                                    'y': None,
                                    "local_image": None if not model_kwargs[0][0].__contains__('local_image') else model_kwargs[0][0]["local_image"][:, :, ind, :, :],
                                    "coarse_image": None if not model_kwargs[0][0].__contains__('coarse_image') else model_kwargs[0][0]["coarse_image"][:, :, ind, :, :],
                                    'image':  None if not model_kwargs[0][0].__contains__('image') else model_kwargs[0][0]["image"].repeat(bs_context, 1, 1),
                                    'dwpose':  None if not model_kwargs[0][0].__contains__('dwpose') else model_kwargs[0][0]["dwpose"][:, :, pose_ind, :, :],
                                    'randomref':  None if not model_kwargs[0][0].__contains__('randomref') else model_kwargs[0][0]["randomref"][:, :, ind, :, :],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
            
            model_kwargs_pre_nc = [{
                                    'y': None,
                                    "local_image": None if not model_kwargs[1][0].__contains__('local_image') else model_kwargs[1][0]["local_image"][:, :, ind, :, :],
                                    'image':  None if not model_kwargs[1][0].__contains__('image') else model_kwargs[1][0]["image"].repeat(bs_context, 1, 1),
                                    'dwpose':  None if not model_kwargs[1][0].__contains__('dwpose') else model_kwargs[1][0]["dwpose"][:, :, pose_ind, :, :],
                                    'randomref':  None if not model_kwargs[1][0].__contains__('randomref') else model_kwargs[1][0]["randomref"][:, :, ind, :, :],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
            
            model_kwargs_tune_c = [{
                                    'y': None,
                                    "local_image": None if not model_kwargs[2][0].__contains__('local_image') else model_kwargs[2][0]["local_image"][:, :, ind, :, :],
                                     "coarse_image": None if not model_kwargs[2][0].__contains__('coarse_image') else model_kwargs[2][0]["coarse_image"][:, :, ind, :, :],
                                    'image':  None if not model_kwargs[2][0].__contains__('image') else model_kwargs[2][0]["image"].repeat(bs_context, 1, 1),
                                    'dwpose':  None if not model_kwargs[2][0].__contains__('dwpose') else model_kwargs[2][0]["dwpose"][:, :, pose_ind, :, :],
                                    'randomref':  None if not model_kwargs[2][0].__contains__('randomref') else model_kwargs[2][0]["randomref"][:, :, ind, :, :],
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
            
            if (coarse_seg is not None) and (coarse_seg_noc is not None):
                model_kwargs_tune_nc[0]['coarse_image'][:,:,0:context_overlap,:,:] = coarse_seg_noc[:,:,-context_overlap:,:]
                model_kwargs_tune_c[0]['coarse_image'][:,:,0:context_overlap,:,:] = coarse_seg_noc[:,:,-context_overlap:,:]
                '''I = model_kwargs_tune_nc[0]['coarse_image']
                frames = []
                for i in range(I.shape[2]):
                    frame = I[0][:3,i].permute(1, 2, 0).cpu().numpy()  # 转换为 (h, w, c) 格式
                    frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)
                imageio.mimsave(str(current_start)+'_coarse.gif', frames, duration=0.1)'''

    
            #generate first clip 并返回 coarse_image 和 xt
            seg, coarse_seg_noc, coarse_seg = self.ddim_sample_blend_loop(
                    noise=noise_one,
                    models=models, 
                    model_kwargs=[model_kwargs_tune_nc, model_kwargs_pre_nc, model_kwargs_tune_c],
                    guide_scale=guide_scale,
                    ddim_timesteps=ddim_timesteps,
                    eta=0.0,
                    mask=mask_seg,
                    vae=vae,
                    tau_alpha=tau_alpha,
                    tau_beta=tau_beta,
                    return_coarse=True,
                    #keep_first=keep_first
                    )
            # b c f h w
            #seg b c f h w # ind list tenosr
            #video_data.append(seg[:,:,context_overlap:,:,:] if current_start > 0 else seg)
            current_start += (context_size - context_overlap)

            video_data_all[:, :, ind, :, :] = video_data_all[:, :, ind, :, :] + seg
            counter[:, :, ind] = counter[:, :, ind] + 1
      
            if current_end == total_frames:
                break
        
        #video_data = torch.cat(video_data, dim=2)
        video_data = video_data_all / counter
        return video_data
    

    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    

    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt
    

    def loss(self, x0, t, model, model_kwargs={}, noise=None, weight = None, use_div_loss= False):

        # noise = torch.randn_like(x0) if noise is None else noise # [80, 4, 8, 32, 32]
        noise = self.sample_loss(x0, noise)
        xt = self.q_sample(x0, t, noise=noise)
        # compute loss
        out = model(xt, self._scale_timesteps(t), **model_kwargs)
        # VLB for variation
        loss_vlb = 0.0
        # MSE/L1 for x0/eps
        # target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
        target = {
            'eps': noise, 
            'x0': x0, 
            'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0], 
            'v':_i(self.sqrt_alphas_cumprod, t, xt) * noise - _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * x0}[self.mean_type]
        

        loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2).abs()
        loss = loss.flatten(1).mean(dim=1)


        loss = loss + loss_vlb
        
        return loss



    
    def variational_lower_bound(self, x0, xt, t, model, model_kwargs={}, clamp=None, percentile=None):
        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2)
        kl = kl.flatten(1).mean(dim=1) / math.log(2.0)
        
        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(x0, mean=mu2, log_scale=0.5 * log_var2)
        nll = nll.flatten(1).mean(dim=1) / math.log(2.0)

        # NLL for p(x0 | x1) and KL otherwise
        vlb = torch.where(t == 0, nll, kl)
        return vlb, x0
    
    @torch.no_grad()
    def variational_lower_bound_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None):
        r"""Compute the entire variational lower bound, measured in bits-per-dim.
        """
        # prepare input and output
        b = x0.size(0)
        metrics = {'vlb': [], 'mse': [], 'x0_mse': []}

        # loop
        for step in torch.arange(self.num_timesteps).flip(0):
            # compute VLB
            t = torch.full((b, ), step, dtype=torch.long, device=x0.device)
            # noise = torch.randn_like(x0)
            noise = self.sample_loss(x0)
            xt = self.q_sample(x0, t, noise)
            vlb, pred_x0 = self.variational_lower_bound(x0, xt, t, model, model_kwargs, clamp, percentile)

            # predict eps from x0
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)

            # collect metrics
            metrics['vlb'].append(vlb)
            metrics['x0_mse'].append((pred_x0 - x0).square().flatten(1).mean(dim=1))
            metrics['mse'].append((eps - noise).square().flatten(1).mean(dim=1))
        metrics = {k: torch.stack(v, dim=1) for k, v in metrics.items()}

        # compute the prior KL term for VLB, measured in bits-per-dim
        mu, _, log_var = self.q_mean_variance(x0, t)
        kl_prior = kl_divergence(mu, log_var, torch.zeros_like(mu), torch.zeros_like(log_var))
        kl_prior = kl_prior.flatten(1).mean(dim=1) / math.log(2.0)

        # update metrics
        metrics['prior_bits_per_dim'] = kl_prior
        metrics['total_bits_per_dim'] = metrics['vlb'].sum(dim=1) + kl_prior
        return metrics

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()





@DIFFUSION.register_class()
class DiffusionDDIMLong(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 **kwargs):

        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # v
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength 

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise


    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    @torch.no_grad()
    def visualize_x(self, vae, x, t, name):
        vae.cuda()
        coarse_I = 1. / 0.18215   * x
        b, c, f, h, w= coarse_I.shape
        coarse_I = rearrange(coarse_I, 'b c f h w -> (b f) c h w') #16,c,h,w
        coarse_I_list = torch.chunk(coarse_I, coarse_I.shape[0]//4, dim=0)
        decode_data = []
        for vd_data in coarse_I_list:
            gen_frames = vae.decode(vd_data.float())
            decode_data.append(gen_frames)
        coarse_I = torch.cat(decode_data, dim=0)
        coarse_I = rearrange(coarse_I, '(b f) c h w -> b c f h w', b=b, f=f) #(-1,1)
        coarse_I = (coarse_I.clamp(-1,1) + 1)/2 #(0,1)
        frames = []
        for i in range(coarse_I.shape[2]):
            frame = coarse_I[0][:3,i].permute(1, 2, 0).cpu().numpy()  # 转换为 (h, w, c) 格式
            frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        os.makedirs('visualization', exist_ok=True)
        imageio.mimsave(os.path.join('visualization',str(t)+'_'+name+'.gif'), frames, duration=0.1)
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, context_size=32, context_stride=1, context_overlap=4, context_batch_size=1):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        noise = xt
        context_queue = list(
                    context_scheduler(
                        0,
                        31,
                        noise.shape[2],
                        context_size=context_size,
                        context_stride=1,
                        context_overlap=context_overlap,
                    )
                )
        context_step = min(
                    context_stride, int(np.ceil(np.log2(noise.shape[2] / context_size))) + 1
                )
        # replace the final segment to improve temporal consistency
        num_frames = noise.shape[2]
        context_queue[-1] = [
                e % num_frames
                for e in range(num_frames - context_size * context_step, num_frames, context_step)
            ]
        
        import math
        # context_batch_size = 1
        num_context_batches = math.ceil(len(context_queue) / context_batch_size)
        global_context = []
        for i in range(num_context_batches):
            global_context.append(
                context_queue[
                    i * context_batch_size : (i + 1) * context_batch_size
                ]
            )
        noise_pred = torch.zeros_like(noise)
        noise_pred_uncond = torch.zeros_like(noise)
        counter = torch.zeros(
                    (1, 1, xt.shape[2], 1, 1),
                    device=xt.device,
                    dtype=xt.dtype,
                )
        
        for i_index, context in enumerate(global_context):
            
            
            latent_model_input = torch.cat([xt[:, :, c] for c in context])
            bs_context = len(context)
            
            model_kwargs_new = [{
                                    'y': None,
                                    "local_image": None if not model_kwargs[0].__contains__('local_image') else torch.cat([model_kwargs[0]["local_image"][:, :, c] for c in context]),
                                    'image':  None if not model_kwargs[0].__contains__('image') else model_kwargs[0]["image"].repeat(bs_context, 1, 1),
                                    'dwpose':  None if not model_kwargs[0].__contains__('dwpose') else torch.cat([model_kwargs[0]["dwpose"][:, :, [0]+[ii+1 for ii in c]] for c in context]),
                                    'randomref':  None if not model_kwargs[0].__contains__('randomref') else torch.cat([model_kwargs[0]["randomref"][:, :, c] for c in context]),
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
            
            if guide_scale is None:
                out = model(latent_model_input, self._scale_timesteps(t), **model_kwargs)
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + out
                    counter[:, :, c] = counter[:, :, c] + 1
            else:
                # classifier-free guidance
                # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
                # assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
                y_out = model(latent_model_input, self._scale_timesteps(t).repeat(bs_context), **model_kwargs_new[0])
                u_out = model(latent_model_input, self._scale_timesteps(t).repeat(bs_context), **model_kwargs_new[1])
                dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + y_out[j:j+1]
                    noise_pred_uncond[:, :, c] = noise_pred_uncond[:, :, c] + u_out[j:j+1]
                    counter[:, :, c] = counter[:, :, c] + 1
                
        noise_pred = noise_pred / counter
        noise_pred_uncond = noise_pred_uncond / counter
        out = torch.cat([
                    noise_pred_uncond[:, :dim] + guide_scale * (noise_pred[:, :dim] - noise_pred_uncond[:, :dim]),
                    noise_pred[:, dim:]], dim=1) # guide_scale=2.5

        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)
        
        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'v':
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_size=32, context_stride=1, context_overlap=4, context_batch_size=1):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """

        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale, context_size, context_stride, context_overlap, context_batch_size)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    

    @torch.no_grad()
    def ddim_sample_condition(self, xt, t, models, model_kwargs, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, 
                              eta=0.0, context_size=32, context_stride=1, context_overlap=4, context_batch_size=1,
                              mask=None, vae=None, inj_start=0, inj_stop=None):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps
        
        model_pre, model_tune = models
        # predict distribution of p(x_{t-1} | x_t)
        if t > inj_start:
            _, _, _, x0 = self.p_mean_variance(xt, t, model_tune, model_kwargs[0], clamp, percentile, guide_scale, context_size, context_stride, context_overlap, context_batch_size)
        else:
            _, _, _, x0_our = self.p_mean_variance(xt, t, model_tune, model_kwargs[0], clamp, percentile, guide_scale, context_size, context_stride, context_overlap, context_batch_size)
            #self.visualize_x(vae, x0_our, t, 'our')
            _, _, _, x0_pre = self.p_mean_variance(xt, t, model_pre, model_kwargs[1], clamp, percentile, guide_scale, context_size, context_stride, context_overlap, context_batch_size)
            #self.visualize_x(vae, x0_pre, t, 'pre')
            x0 = (1-mask) * x0_our +  mask * x0_pre
            #self.visualize_x(vae, x0, t, 'mix')
        
            if t == inj_stop:
                print('update the coarse_condition:',t)
    
                vae.cuda()
                coarse_I = 1. / 0.18215   * x0
                b, c, f, h, w= coarse_I.shape
                coarse_I = rearrange(coarse_I, 'b c f h w -> (b f) c h w') #16,c,h,w
                coarse_I_list = torch.chunk(coarse_I, coarse_I.shape[0]//4, dim=0)
                decode_data = []
                for vd_data in coarse_I_list:
                    gen_frames = vae.decode(vd_data.float())
                    decode_data.append(gen_frames)
                coarse_I = torch.cat(decode_data, dim=0)
                coarse_I = rearrange(coarse_I, '(b f) c h w -> b c f h w', b=b, f=f) #(-1,1)
                coarse_I = (coarse_I.clamp(-1,1) + 1)/2 #(0,1)

                coarse_org, mask_org = model_kwargs[0][0]['coarse_image'][:,:3], model_kwargs[0][0]['coarse_image'][:,3:]
                
                coarse_I = mask_org * coarse_I + (1 - mask_org) * coarse_org 
                mask_new = torch.zeros_like(mask_org)
                coarse_I = torch.cat([coarse_I, mask_new], dim=1)

                model_kwargs[0][0]['coarse_image'] = coarse_I
                model_kwargs[2][0]['coarse_image'] = coarse_I

                # b c f h w
                frames = []
                for i in range(coarse_I.shape[2]):
                    frame = coarse_I[0][:3,i].permute(1, 2, 0).cpu().numpy()  # 转换为 (h, w, c) 格式
                    frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)
                imageio.mimsave(str(t)+'_coarse.gif', frames, duration=0.1)

        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    
    @torch.no_grad()
    def ddim_sample_loop(self, noise, context_size, context_stride, context_overlap, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_batch_size=1, vae=None):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        from tqdm import tqdm
        
        for step in tqdm(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, x0 = self.ddim_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta, context_size=context_size, context_stride=context_stride, context_overlap=context_overlap, context_batch_size=context_batch_size)
            #self.visualize_x(vae, x0, t, 'pre-org')
        return xt
    

    '''@torch.no_grad()
    def ddim_sample_loop_change(self, noise, context_size, context_stride, context_overlap, models, model_kwargs, clamp=None, percentile=None, 
                                condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_batch_size=1, mask=None, vae=None, inj_start=0.1, inj_stop=0.2):
        # prepare input
        b = noise.size(0)
        xt_coarse = noise
        xt = noise.clone()

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        #inj_start = int(self.num_timesteps * inj_start)
        #inj_stop = int(self.num_timesteps * inj_stop)
        from tqdm import tqdm
        
        for step in tqdm(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt_coarse.device)
            xt_coarse, _ = self.ddim_sample_condition(xt_coarse, t, models, model_kwargs, clamp, percentile, condition_fn, 
                                                     guide_scale, ddim_timesteps, eta, context_size=context_size, context_stride=context_stride, 
                                                     context_overlap=context_overlap, context_batch_size=context_batch_size,
                                                     mask=mask, vae=vae, inj_start=inj_start, inj_stop=inj_stop)
            if step == inj_stop:
                break
        print('Denosing with new conditions')
        for step in tqdm(steps):
           
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_sample(xt, t, models[1], model_kwargs[2], clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta, context_size=context_size, context_stride=context_stride, context_overlap=context_overlap, context_batch_size=context_batch_size)
        return xt'''
    

    
    
    @torch.no_grad()
    def ddim_reverse_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            torch.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)
        
        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt
    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def loss(self, x0, t, model, model_kwargs={}, noise=None, weight = None, use_div_loss= False, loss_mask=None):

        # noise = torch.randn_like(x0) if noise is None else noise # [80, 4, 8, 32, 32]
        noise = self.sample_loss(x0, noise)

        xt = self.q_sample(x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model, model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1', 'rescaled_l1']: # self.loss_type: mse
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']: # self.var_type: 'fixed_small'
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            # target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            target = {
                'eps': noise, 
                'x0': x0, 
                'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0], 
                'v':_i(self.sqrt_alphas_cumprod, t, xt) * noise - _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * x0}[self.mean_type]
            if loss_mask is not None:
                loss_mask = loss_mask[:, :, 0, ...].unsqueeze(2)  # just use one channel (all channels are same)
                loss_mask = loss_mask.permute(0, 2, 1, 3, 4)  # b,c,f,h,w 
                # use masked diffusion
                loss = (out * loss_mask - target * loss_mask).pow(1 if self.loss_type.endswith('l1') else 2).abs().flatten(1).mean(dim=1)
            else:
                loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2).abs().flatten(1).mean(dim=1)
            if weight is not None:
                loss = loss*weight   

            # div loss
            if use_div_loss and self.mean_type == 'eps' and x0.shape[2]>1:
                 
                # derive  x0
                x0_ = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out


                # ncfhw, std on f
                div_loss = 0.001/(x0_.std(dim=2).flatten(1).mean(dim=1)+1e-4)
                # print(div_loss,loss)
                loss = loss+div_loss

            # total loss
            loss = loss + loss_vlb
        elif self.loss_type in ['charbonnier']:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            loss = torch.sqrt((out - target)**2 + self.epsilon)
            if weight is not None:
                loss = loss*weight
            loss = loss.flatten(1).mean(dim=1)
            
            # total loss
            loss = loss + loss_vlb
        return loss

    def variational_lower_bound(self, x0, xt, t, model, model_kwargs={}, clamp=None, percentile=None):
        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2)
        kl = kl.flatten(1).mean(dim=1) / math.log(2.0)

        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(x0, mean=mu2, log_scale=0.5 * log_var2)
        nll = nll.flatten(1).mean(dim=1) / math.log(2.0)

        # NLL for p(x0 | x1) and KL otherwise
        vlb = torch.where(t == 0, nll, kl)
        return vlb, x0
    
    @torch.no_grad()
    def variational_lower_bound_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None):
        r"""Compute the entire variational lower bound, measured in bits-per-dim.
        """
        # prepare input and output
        b = x0.size(0)
        metrics = {'vlb': [], 'mse': [], 'x0_mse': []}

        # loop
        for step in torch.arange(self.num_timesteps).flip(0):
            # compute VLB
            t = torch.full((b, ), step, dtype=torch.long, device=x0.device)
            # noise = torch.randn_like(x0)
            noise = self.sample_loss(x0)
            xt = self.q_sample(x0, t, noise)
            vlb, pred_x0 = self.variational_lower_bound(x0, xt, t, model, model_kwargs, clamp, percentile)

            # predict eps from x0
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)

            # collect metrics
            metrics['vlb'].append(vlb)
            metrics['x0_mse'].append((pred_x0 - x0).square().flatten(1).mean(dim=1))
            metrics['mse'].append((eps - noise).square().flatten(1).mean(dim=1))
        metrics = {k: torch.stack(v, dim=1) for k, v in metrics.items()}

        # compute the prior KL term for VLB, measured in bits-per-dim
        mu, _, log_var = self.q_mean_variance(x0, t)
        kl_prior = kl_divergence(mu, log_var, torch.zeros_like(mu), torch.zeros_like(log_var))
        kl_prior = kl_prior.flatten(1).mean(dim=1) / math.log(2.0)

        # update metrics
        metrics['prior_bits_per_dim'] = kl_prior
        metrics['total_bits_per_dim'] = metrics['vlb'].sum(dim=1) + kl_prior
        return metrics

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()



def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def context_scheduler(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


