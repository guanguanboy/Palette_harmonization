import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

from tqdm import tqdm
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')

    args = parser.parse_args()

    harmonized_paths = []
    real_paths = []

    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220505_145224/results/test/0/'
    output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220505_162755/results/test/0/'
    files = '/data1/liguanlin/Datasets/iHarmony/Hday2night/Hday2night_test.txt'
    with open(files,'r') as f:
            for line in f.readlines():
                name_str = line.rstrip()
                harmonized_img_name = 'Out_' + name_str
                harmonized_path = os.path.join(output_path, harmonized_img_name)
                real_img_name = 'In_' + name_str
                real_path = os.path.join(output_path, real_img_name)

                real_paths.append(real_path)
                harmonized_paths.append(harmonized_path)

    mse_scores = 0
    psnr_scores = 0
    ssim_scores = 0
    image_size = 256

    count = 0

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')

        harmonized_np = np.array(harmonized, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)

        mse_score = mse(harmonized_np, real_np)
        psnr_score = psnr(real_np, harmonized_np, data_range=255)
        ssim_score = ssim(real_np, harmonized_np, data_range=255, multichannel=True)

        psnr_scores += psnr_score
        mse_scores += mse_score
        ssim_scores += ssim_score

    mse_scores_mu = mse_scores/count
    psnr_scores_mu = psnr_scores/count
    ssim_scores_mu = ssim_scores/count

    print(count)
    mean_sore = "MSE %0.2f | PSNR %0.2f | ssim %0.3f" % (mse_scores_mu, psnr_scores_mu, ssim_scores_mu)
    print(mean_sore)    