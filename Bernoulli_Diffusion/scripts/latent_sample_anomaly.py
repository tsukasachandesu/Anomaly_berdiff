"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from skimage import exposure
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append('./Binary_AE')
sys.path.append('./Bernoulli_Diffusion')
import csv
import numpy as np
import nibabel
import torch as torch
import torch.distributed as dist
from guided_diffusion.train_util import visualize
import scipy
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
import torchvision.transforms.functional as F
torch.manual_seed(0)
import random
random.seed(0)

import sys
import argparse


from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop
import numpy as np
from PIL import Image
def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def main():

    H = get_sampler_hparams()

    args, unknown = create_argparser().parse_known_args()
    print('args', args)

    datal = load_data(
        data_dir=H.data_dir,
        batch_size=1,
        image_size=H.image_size,
    )
    val_loader = iter(datal)
    dist_util.setup_dist()
    print('device',dist_util.dev() )
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print('loaded model', args.model_path)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('unet', count_parameters(model))

    a,b = diffusion.p_sample_loop_anomaly(model,(1, 128,32, 32))
    img_array = (a* 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("/content/sample_data/5.jpg")

    logger.log("sampling...")
    args.num_samples = 1
    logger.log(args.num_samples)

    k=0
    while k < args.num_samples:
        k+=1
        if args.dataset=='brats':
            data = next(val_loader)
            batch=data
        print('batch', batch.shape)

        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop_anomaly if not args.use_ddim else diffusion.ddim_sample_loop_anomaly
        )

        img=batch.cuda()

        code = img

        sample, mask= sample_fn(
            model,
            (args.batch_size, 128,32, 32),
            code,
            prob_threshold=args.prob_threshold,
            noise_level=args.noise_level,
            model_kwargs=model_kwargs,
        )

        torch.cuda.synchronize()

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        num_samples=1038,
        batch_size=1,
        prob_threshold=0.5,
        noise_level=200,
        data_dir="./data/brats/validation",
        use_ddim=True,
        model_path='./results/brats000000.pt',
        use_fp16=False,
        img_channels=4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        fp16_scale_growth=1e-3,
        dataset='brats',
        ae_load_dir='',
        ae_load_step=260000,
        sampler="bld",
        codebook_size=64,
        nf=32,
        img_size=256,
        latent_shape=[1, 128, 128],
        n_channels=4,
        ch_mult=[1, 2],
        mean_type="epsilon"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
