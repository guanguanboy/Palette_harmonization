from gaussian_diffusion import get_named_beta_schedule
from respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )

if __name__ == "__main__":
    dpm = create_gaussian_diffusion(steps=1000, noise_schedule='squaredcos_cap_v2', timestep_respacing='100')
    print(dpm)