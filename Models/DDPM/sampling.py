# sampling.py
import torch
from model import Diffuser
from train import ExponentialMovingAverage
from torchvision.utils import save_image
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from MNISTDiffusion")
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--conditional', action='store_true', help='conditional training')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    args=parse_args()

    device = "cpu" if args.cpu else "cuda"
    CONDITIONAL = True if args.conditional else False
    ckpt_path = "./saves/conditional.pt" if CONDITIONAL else "./checkpoint/unconditional_model.pt"
    SPECIFIC_NUMBER_LIST = torch.full((args.n_samples,), args.number).to(device) if CONDITIONAL else None
    # SPECIFIC_NUMBER_LIST = torch.tensor([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8]).to(device) if CONDITIONAL else None
    SAMPLE_NUM = args.n_samples

    print("loading...")
    model = Diffuser(
        timesteps=1000,
        label=9,
        time_embedding_dim=256,
        image_size=32,
        in_channels=1,
        base_dim=128,
        dim_mults=[2, 4, 8]
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - model.alphas)
    model_ema.eval()
    print("loaded successful!")
    print("simpling...")
    samples=model_ema.module.sampling(SAMPLE_NUM, device=device,l=SPECIFIC_NUMBER_LIST)
    save_image(samples, f'{"conditional_" if CONDITIONAL else "unconditional_"}sampling.png', nrow=int(math.sqrt(SAMPLE_NUM)))
    print("sampling successful!")
