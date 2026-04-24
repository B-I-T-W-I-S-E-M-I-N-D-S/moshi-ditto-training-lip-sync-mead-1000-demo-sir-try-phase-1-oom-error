import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from tqdm import tqdm

from .utils import extract, make_beta_schedule


def identity(t, *args, **kwargs):
    return t


class MotionDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        use_ema=False,
        part_w_dict=None,
        use_last_frame_loss=False,
        use_reg_loss=False,
        dim_ws=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model

        self.ema = None
        self.master_model = None

        self.cond_drop_prob = cond_drop_prob

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        self.part_w_dict = part_w_dict
        self.use_last_frame_loss = use_last_frame_loss
        self.use_reg_loss = use_reg_loss

        if self.part_w_dict is None:
            # self.part_w_dict = {'mtn': (0, -1, 1)}
            self.part_w_dict = {"scale": [0, 1, 1], "pitch": [1, 67, 1], "yaw": [67, 133, 1], "roll": [133, 199, 1], "t": [199, 202, 1], "exp": [202, 265, 1]}

        if dim_ws is not None:
            self.register_buffer("dim_ws", torch.from_numpy(dim_ws))
        else:
            self.dim_ws = None
        
    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond_frame, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond_frame, cond, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def ddim_sample(self, shape, cond_frame, cond, last_half=None, noise=None, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if noise is not None:
            x = noise.to(device)
        else:
            x = torch.randn(shape, device = device)
        cond_frame = cond_frame.to(device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond_frame, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond_frame, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond_frame, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        def _get_pva_loss(pred, gt, part_w_dict, use_last_frame_loss=False, use_reg_loss=False, dim_ws=None):
                
            def _part_loss(s, e, w, rw=0):
                if s <= 0:
                    s = 0
                if e <= 0:
                    e = gt.shape[-1]

                dim_w = w
                if dim_ws is not None:
                    dim_w = dim_ws[s:e][None, None] * w    # [1, 1, dim]

                p1 = pred[..., s:e]
                p2 = gt[..., s:e]
                
                v1 = p1[:, 1:] - p1[:, :-1]
                v2 = p2[:, 1:] - p2[:, :-1]
                
                a1 = v1[:, 1:] - v1[:, :-1]
                a2 = v2[:, 1:] - v2[:, :-1]

                _p_loss = self.loss_fn(p1, p2, reduction="none") * dim_w    # [b, L, dim]
                _v_loss = self.loss_fn(v1, v2, reduction="none") * dim_w
                _a_loss = self.loss_fn(a1, a2, reduction="none") * dim_w

                _p_loss = _p_loss.mean()
                _v_loss = _v_loss.mean()
                _a_loss = _a_loss.mean()

                _l_loss = 0
                if use_last_frame_loss:
                    p0 = p1[:, 0:1]    # [b, 1, dim]
                    gt0 = cond_frame[..., s:e][:, None]
                    _l_loss = self.loss_fn(p0, gt0, reduction="none") * dim_w
                    _l_loss = _l_loss.mean()

                _r_loss = 0
                if use_reg_loss:
                    _r_loss = torch.abs(p1).mean() * rw
                
                return _p_loss, _v_loss, _a_loss, _l_loss, _r_loss
                
            loss_dict = {}
            for k, (s, e, w) in part_w_dict.items():
                if k == 'scale':
                    rw = 0
                else:
                    rw = 1e-4

                _p_loss, _v_loss, _a_loss, _l_loss, _r_loss = _part_loss(s, e, w, rw)
                loss_dict[f'{k}_P'] = _p_loss.mean()
                loss_dict[f'{k}_V'] = _v_loss.mean()
                loss_dict[f'{k}_A'] = _a_loss.mean()
                if use_last_frame_loss:
                    loss_dict[f'{k}_L'] = _l_loss.mean()
                if use_reg_loss:
                    loss_dict[f'{k}_R'] = _r_loss.mean()
            
            return loss_dict
        
        if self.part_w_dict:
            loss_dict = _get_pva_loss(model_out, target, self.part_w_dict, self.use_last_frame_loss, self.use_reg_loss, dim_ws=self.dim_ws)
            total_loss = sum(loss_dict.values())
            return total_loss, loss_dict, x_recon
        else:
            raise NotImplementedError()

    def loss(self, x, cond_frame, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        total_loss, loss_dict, x_recon = self.p_losses(x, cond_frame, cond, t)
        return total_loss, loss_dict, x_recon

    def forward(self, x, cond_frame, cond, t_override=None):
        total_loss, loss_dict, x_recon = self.loss(x, cond_frame, cond, t_override)
        return total_loss, loss_dict, x_recon

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample(
        self,
        shape,
        cond_frame,
        cond,
        normalizer,
        epoch,
        render_out,
        last_half=None,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond_frame,
                    cond,
                    last_half=last_half,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        if render_out is None:
            return samples

        os.makedirs(render_out, exist_ok=True)
        for i in range(samples.shape[0]):
            np.save(f"{render_out}/{epoch}_{os.path.basename(name[i])[:-4]}.npy", samples[i].numpy())
            
        return