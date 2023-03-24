import math
from sqlalchemy import null
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torchvision.transforms import Resize
import torch.nn.functional as F

from models.gaussian_diffusion import get_named_beta_schedule
from models.respace import SpacedDiffusion, space_timesteps
from models.lap_pyr_model import Lap_Pyramid_Conv,Trans_high,Trans_high_masked_residual

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

NUM_CLASSES = 1000

from .guided_diffusion_modules.unet_improved import UNetModel

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    # how unet constructed
    if image_size == 256: 
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=6,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6), #如果学习sigma的话，输出是6个通道，前三个是通道预测eps噪声，后三个通道预测方差
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet_modified import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net
        elif module_name == 'improved_laplacian':
            from .guided_diffusion_modules.unet_modified_laplacian import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        self.beta_schedule = beta_schedule
        self.num_timesteps = beta_schedule['train']['n_timestep']

        #print(self.num_timesteps)
        if not beta_schedule['test']['is_test']:
            self.time_step_respacing = self.num_timesteps
            self.spaced_dpm = self._create_gaussian_diffusion(steps=self.num_timesteps, noise_schedule='squaredcos_cap_v2')
        else:
            self.time_step_respacing = beta_schedule['test']['time_step_respacing']
            self.spaced_dpm = self._create_gaussian_diffusion(steps=self.num_timesteps, noise_schedule='squaredcos_cap_v2', timestep_respacing=str(self.time_step_respacing))

        self.is_ddim = True

        self.lap_pyramid = Lap_Pyramid_Conv(num_high=2, device=torch.device('cuda'))

        for param in self.lap_pyramid.parameters():
            param.requires_grad = False

        self.trans_high = Trans_high_masked_residual(num_residual_blocks=3, num_high=2)

        #self.refine_net = RefineNet
        #self.parameterization = "eps" 
        #另一个值是x0
        self.parameterization = "x0"

    def _create_gaussian_diffusion(self, steps, noise_schedule, timestep_respacing=''):
        betas = get_named_beta_schedule(noise_schedule, steps)
        if not timestep_respacing:
            timestep_respacing = [steps]
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
        )

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        #gammas_prev = np.append(1., gammas[:-1])


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        """
        pass

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn


    def q_sample(self, y_0, time_step, noise=None): #计算扩散过程中任意时刻y_t的采样值，直接套公式,采样得到一张噪声图片
        return self.spaced_dpm.q_sample(x_start=y_0, t=time_step, noise=noise)

    def get_recon_res(self, pyr, mask, model_output):
        fake_B_low = model_output
        real_A_up = F.interpolate(pyr[-1], size=(pyr[-2].shape[2], pyr[-2].shape[3]))
        fake_B_up = F.interpolate(fake_B_low, size=(pyr[-2].shape[2], pyr[-2].shape[3]))
        mask = F.interpolate(mask, size=(pyr[-2].shape[2], pyr[-2].shape[3]))
        high_with_low = torch.cat([pyr[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, mask, pyr, fake_B_low)

        enlarged_output = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        return enlarged_output

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None): #从y_t采样t时刻的重构值
        # Pack the tokens together into model kwargs. 用字典来保存模型参数，提高了模型接口的可扩展性
        model_kwargs = dict(

            #mask=mask,#torch.Size([2, 128])

            # Masked inpainting image
            y_cond=y_cond,
            #inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
        )

        #需要在这里把相关的参数给整理好
        #out = self.spaced_dpm.p_sample(model=self.denoise_fn, x=y_t, t=t, clip_denoised=clip_denoised, denoised_fn=None, cond_fn=None,model_kwargs=model_kwargs)

        if True == self.is_ddim:
            out = self.spaced_dpm.ddim_sample_dp_laplacian(model=self.denoise_fn, x=y_t, t=t, clip_denoised=clip_denoised, denoised_fn=None, cond_fn=None,model_kwargs=model_kwargs)

        else:
            out = self.spaced_dpm.p_sample_dp(model=self.denoise_fn, x=y_t, t=t, clip_denoised=clip_denoised, denoised_fn=None, cond_fn=None,model_kwargs=model_kwargs)
        
        image = out["sample"]

        return image

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8): #采样过程，类似于IDDPM中的源码p_sample_loop
        #这里也需要把相关图像进行分解
        pyr = self.lap_pyramid.pyramid_decom(y_cond)
        y_cond_down = pyr[-1]
        #对mask也做一个下采样
        mask_down = torch.nn.functional.interpolate(mask, size=y_cond_down.shape[-2:])

        b, *_ = y_cond.shape

        assert self.time_step_respacing > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.time_step_respacing//sample_num)
        
        y_t = default(y_cond_down, lambda: torch.randn_like(y_cond_down))
        #ret_arr = torch.nn.functional.interpolate(y_t, size=y_cond.shape[-2:]) #保存下来的图像
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.time_step_respacing)), desc='sampling loop time step', total=self.time_step_respacing):
            t = torch.full((b,), i, device=y_cond_down.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond_down) #将y_t作为下一个迭代的输入来生成新的y_t #会在p_sample调用函数。
            if mask_down is not None:
                y_t = y_cond_down*(1.-mask_down) + mask_down*y_t #得到y_t之后，将y_t作为下一个sample 生成的输入
                #pyr[-1] = y_t
                y_restored = self.get_recon_res(pyr, mask_down, y_t)
                y_restored = y_cond * (1. - mask) + mask*y_restored

            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_restored, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None): #参数顺序，真实图片y_0，合成图片(条件图片y_cond)以及mask
        # sampling from p(gammas)该函数的输出是loss，可以调用IDDPM中的compute_losses函数来实现。

        #这里需要做一个laplace的下采样
        pyr = self.lap_pyramid.pyramid_decom(y_cond)
        y_cond_down = pyr[-1].detach()

        
        #对mask也做一个下采样
        mask_down = torch.nn.functional.interpolate(mask, size=y_cond_down.shape[-2:])

        ###构造t
        b, *_ = y_0.shape

        #对y_0做下采样
        #y0_down = torch.nn.functional.interpolate(y_0, size=y_cond_down.shape[-2:])
        y_0_pyr = self.lap_pyramid.pyramid_decom(y_0)
        y0_down = y_0_pyr[-1].detach()
        
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机生成一个时间点

        noise_down = default(noise, lambda: torch.randn_like(y_cond_down))


        #构造可变参数
        model_kwargs = dict(

            mask= mask_down,#torch.Size([2, 128])

            # Masked inpainting image
            y_cond=y_cond_down,
            #inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
            noise=noise_down,
        )
        

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = y_0
        else:
            raise NotImplementedError()

        #希望这里能返回预测好的x0
        loss, model_output = self.spaced_dpm.get_training_losses_and_x0(self.denoise_fn, y0_down, t, model_kwargs=model_kwargs)

        
        model_output_transformed = model_output*mask_down + y_cond_down*(1. - mask_down)

        #先使用laplance重建对模型进行重建，
        #pyr[-1] = model_output
        #enlarged_output = self.lap_pyramid.pyramid_recons(pyr)

        enlarged_output = self.get_recon_res(pyr, mask_down, model_output_transformed)

        #这里再增加一个refinement模块，对模型的输出进行
        #

        #for i in range(len(model_output_list)):
        #    loss += self.loss_fn(mask_resized_list[i]*target_resized_list[i], mask_resized_list[i]*model_output_list[i])
        #loss["ddpm"] = loss["loss"]
        #loss["loss"] += self.loss_fn(mask*target, mask*enlarged_output)
        loss["loss"] = 16 * loss["loss"] + self.loss_fn(mask*target, mask*enlarged_output)
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


if __name__ == "__main__":
    dpm = create_gaussian_diffusion(steps=1000, noise_schedule='linear', timestep_respacing="100")
    print(dpm)