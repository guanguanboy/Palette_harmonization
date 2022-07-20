import torch
from models.guided_diffusion_modules.unet_improved import UNetModel


NUM_CLASSES = 1000

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


if __name__ == "__main__":
    
    model_defaults = dict(
    image_size=256,
    num_channels=128,
    num_res_blocks=3,
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=4,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,)

    unet_model = create_model(**model_defaults)
    print(unet_model)

    input_tensor = torch.randn(1, 6, 256, 256)
    timesteps = torch.tensor([360.])
    output_tensor = unet_model(input_tensor, timesteps)
    print(output_tensor.shape) #torch.Size([1, 6, 256, 256])
