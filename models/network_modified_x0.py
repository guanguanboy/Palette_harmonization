import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torchvision.transforms import Resize
import torch.nn.functional as F
from models.lap_pyr_model import Lap_Pyramid_Conv
#from models.refine_net import RefineNet

def resize_tensor(input_tensor):
    width=input_tensor.shape[2]
    height = input_tensor.shape[3]
    output_tensor = input_tensor
    output_tensor_list = []
    output_tensor_list.append(output_tensor)
    for i in range(3):
        width = width//2
        height = height//2
        tensor = output_tensor
        torch_resize_fun = Resize([width,height])
        output_tensor = torch_resize_fun(tensor)
        output_tensor_list.insert(0, output_tensor)

    return output_tensor_list

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet_modified import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        elif module_name == 'transformer':
            from .transformer_modules.timeswinir import TimeSwinIR
            self.denoise_fn = TimeSwinIR(**unet) #去噪模型是一个u-net

        elif module_name == 'wavelet':
            from .guided_diffusion_modules.unet_wavelet_skip import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net
        elif module_name == 'focal':
            from .guided_diffusion_modules.unet_modified_focal_attn import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net
        elif module_name == 'noise_level_estimation':
            from .guided_diffusion_modules.unet_modified_with_est import UNet
            self.denoise_fn = UNet(**unet)

        self.beta_schedule = beta_schedule

        self.lap_pyramid = Lap_Pyramid_Conv(num_high=1, device=torch.device('cuda'))

        #self.refine_net = RefineNet
        #self.parameterization = "eps" 
        #另一个值是x0
        self.parameterization = "x0" 

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas',
                             to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas',
                             to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas',
                             to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        predicted_noise = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=predicted_noise[-1]) #加-1是为了使用deep supervison

        if self.parameterization == "eps": 
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=predicted_noise[-1]) #加-1是为了使用deep supervison
        elif self.parameterization == "x0":
            y_0_hat = predicted_noise[-1]
        else:
            raise NotImplementedError()

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior( #套公式计算均值和方差
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None): #计算扩散过程中任意时刻y_t的采样值，直接套公式
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None): #从y_t采样t时刻的重构值
        model_mean, model_log_variance = self.p_mean_variance( #计算均值和方差
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t) #生成正太分布的随机量
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8): #采样过程
        #这里也需要把相关图像进行分解
        pyr = self.lap_pyramid.pyramid_decom(y_cond)
        y_cond_down = pyr[-1]
        #对mask也做一个下采样
        mask_down = torch.nn.functional.interpolate(mask, size=y_cond_down.shape[-2:])
        y_0_down = torch.nn.functional.interpolate(y_0, size=y_cond_down.shape[-2:])
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond_down))
        ret_arr = torch.nn.functional.interpolate(y_t, size=y_cond.shape[-2:]) #保存下来的图像
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond_down.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond_down) #将y_t作为下一个迭代的输入来生成新的y_t #会在p_sample调用函数。
            if mask_down is not None:
                y_t = y_0_down*(1.-mask_down) + mask_down*y_t #得到y_t之后，将y_t作为下一个sample 生成的输入
                pyr[-1] = y_t
                y_restored = self.lap_pyramid.pyramid_recons(pyr)
                y_restored = y_0 * (1. - mask) + mask*y_restored

            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_restored], dim=0)
        
        return y_restored, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None): #参数顺序，真实图片y_0，合成图片(条件图片y_cond)以及mask
        # sampling from p(gammas)

        #这里需要做一个laplace的下采样
        pyr = self.lap_pyramid.pyramid_decom(y_cond)
        y_cond_down = pyr[-1]
        #对mask也做一个下采样
        mask_down = torch.nn.functional.interpolate(mask, size=y_cond_down.shape[-2:])

        b, *_ = y_0.shape

        #对y_0做下采样
        y0_down = torch.nn.functional.interpolate(mask, size=y_cond_down.shape[-2:])
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机生成一个时间点
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1) #torch.Size([4, 1]) 4是batch_size

        noise_down = default(noise, lambda: torch.randn_like(y_cond_down))
        
        y_noisy = self.q_sample(
            y_0=y0_down, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_down) #逐渐的给目标图像增加高斯噪声

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = y_0
        else:
            raise NotImplementedError()

        target_resized_list = resize_tensor(target)
        mask_resized_list = resize_tensor(mask)

        loss = 0
        if mask is not None: #如果包含mask，则去噪的时候，将随机噪声y_noisy*mask+真实图片*（1-mask）作为一个输入

            model_output_list = self.denoise_fn(torch.cat([y_cond_down, y_noisy*mask_down+(1.-mask_down)*y0_down], dim=1), sample_gammas)

            #先使用laplance重建对模型进行重建，
            pyr[-1] = model_output_list[-1]
            enlarged_output = self.lap_pyramid.pyramid_recons(pyr)
            #这里再增加一个refinement模块，对模型的输出进行
            #
            
            #for i in range(len(model_output_list)):
            #    loss += self.loss_fn(mask_resized_list[i]*target_resized_list[i], mask_resized_list[i]*model_output_list[i])
            loss += self.loss_fn(mask*target, mask*enlarged_output)
        else:
            model_output_list = self.denoise_fn(torch.cat([y_cond_down, y_noisy], dim=1), sample_gammas)
            #for i in range(len(model_output_list)):
            #    loss += self.loss_fn(mask_resized_list[i]*target_resized_list[i], mask_resized_list[i]*model_output_list[i])
            pyr[-1] = model_output_list[-1]
            enlarged_output = self.lap_pyramid.pyramid_recons(pyr)
            loss += self.loss_fn(mask*target, mask*enlarged_output)

        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


